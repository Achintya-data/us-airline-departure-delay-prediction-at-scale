# scaleout_gbt_job.py
import argparse, time, json
from datetime import datetime

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array

# ---------- Helpers ----------
def now_id():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def timed(label, fn):
    t0 = time.time()
    out = fn()
    dt = time.time() - t0
    print(f"[TIMER] {label}: {dt:.2f}s ({dt/60:.2f}m)")
    return out, dt

def recall_at_topk_fast(df_pred_p1, k=0.05):
    # df_pred_p1 must have: label (0/1), p1 (prob of class 1)
    n = df_pred_p1.count()
    topn = max(1, int(n * k))
    top = df_pred_p1.orderBy(F.desc("p1")).limit(topn)
    tp = top.agg(F.sum(F.col("label").cast("double"))).first()[0] or 0.0
    total_pos = df_pred_p1.agg(F.sum(F.col("label").cast("double"))).first()[0] or 1.0
    return float(tp / total_pos), int(topn), int(n)

def safe_write_json_text(spark, obj, path):
    # Spark text writer requires a SINGLE column called "value"
    payload = json.dumps(obj)
    spark.createDataFrame([(payload,)], ["value"]).coalesce(1).write.mode("overwrite").text(path)

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_base", required=True)     # .../gold/sample_step2_features
    parser.add_argument("--out_base", required=True)      # .../airline/scaling/scaleout
    parser.add_argument("--sample_frac", type=float, default=1.0)
    parser.add_argument("--run_tag", default="")
    parser.add_argument("--shuffle_partitions", type=int, default=200)
    parser.add_argument("--default_parallelism", type=int, default=200)

    # gbt params (keep fixed across scale-out!)
    parser.add_argument("--maxIter", type=int, default=80)
    parser.add_argument("--maxDepth", type=int, default=6)
    parser.add_argument("--stepSize", type=float, default=0.08)
    parser.add_argument("--subsamplingRate", type=float, default=0.8)

    args = parser.parse_args()

    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
    spark.conf.set("spark.default.parallelism", str(args.default_parallelism))

    RUN_ID = now_id()
    run_name = f"run_{RUN_ID}" + (f"_{args.run_tag}" if args.run_tag else "")
    OUT = f"{args.out_base.rstrip('/')}/{run_name}/"
    print("OUT:", OUT)

    # 1) Read gold splits
    def read_split(split):
        return spark.read.parquet(f"{args.gold_base.rstrip('/')}/{split}/")

    (train_df, dt_read_tr) = timed("read train", lambda: read_split("train"))
    (val_df,   dt_read_va) = timed("read val",   lambda: read_split("val"))
    (test_df,  dt_read_te) = timed("read test",  lambda: read_split("test"))

    # 2) Optional sampling (scale-up later)
    if args.sample_frac < 1.0:
        def samp(df): return df.sample(False, args.sample_frac, seed=42)
        (train_df, dt_s_tr) = timed("sample train", lambda: samp(train_df))
        (val_df,   dt_s_va) = timed("sample val",   lambda: samp(val_df))
        (test_df,  dt_s_te) = timed("sample test",  lambda: samp(test_df))
    else:
        dt_s_tr = dt_s_va = dt_s_te = 0.0

    # 3) Keep only what training needs
    # Must exist in your gold: label, features. weight optional.
    cols = train_df.columns
    use_weight = "weight" in cols

    keep = ["label", "features"] + (["weight"] if use_weight else [])
    train_fe = train_df.select(*keep).cache()
    val_fe   = val_df.select(*keep).cache()
    test_fe  = test_df.select(*keep).cache()

    (_, dt_cnt_tr) = timed("count train", lambda: train_fe.count())
    (_, dt_cnt_va) = timed("count val",   lambda: val_fe.count())
    (_, dt_cnt_te) = timed("count test",  lambda: test_fe.count())

    # 4) Train GBT
    def fit():
        gbt = GBTClassifier(
            featuresCol="features",
            labelCol="label",
            weightCol=("weight" if use_weight else None),
            maxIter=args.maxIter,
            maxDepth=args.maxDepth,
            stepSize=args.stepSize,
            subsamplingRate=args.subsamplingRate
        )
        # if no weights, Spark will ignore weightCol=None if we don’t set it
        if not use_weight:
            gbt = GBTClassifier(featuresCol="features", labelCol="label",
                                maxIter=args.maxIter, maxDepth=args.maxDepth,
                                stepSize=args.stepSize, subsamplingRate=args.subsamplingRate)
        return gbt.fit(train_fe)

    (gbt_model, dt_fit) = timed("fit gbt", fit)

    # 5) Predict + evaluate
    e_pr  = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR")
    e_roc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

    def predict_eval(df):
        pred = gbt_model.transform(df).select("label","rawPrediction","probability")
        pred = pred.withColumn("p1", vector_to_array(F.col("probability"))[1]).select("label","rawPrediction","p1")
        return pred.cache()

    (val_pred, dt_pred_va) = timed("predict val",  lambda: predict_eval(val_fe))
    (test_pred,dt_pred_te) = timed("predict test", lambda: predict_eval(test_fe))

    (_, dt_cnt_vp) = timed("count val_pred",  lambda: val_pred.count())
    (_, dt_cnt_tp) = timed("count test_pred", lambda: test_pred.count())

    (val_pr,  dt_val_pr)  = timed("val PR-AUC",  lambda: float(e_pr.evaluate(val_pred)))
    (test_pr, dt_test_pr) = timed("test PR-AUC", lambda: float(e_pr.evaluate(test_pred)))
    (test_roc,dt_test_roc)= timed("test ROC-AUC",lambda: float(e_roc.evaluate(test_pred)))

    (rec_top5, topn, nrows) = recall_at_topk_fast(test_pred.select("label","p1"), k=0.05)

    total_time = (
        dt_read_tr + dt_read_va + dt_read_te +
        dt_s_tr + dt_s_va + dt_s_te +
        dt_cnt_tr + dt_cnt_va + dt_cnt_te +
        dt_fit +
        dt_pred_va + dt_pred_te +
        dt_cnt_vp + dt_cnt_tp +
        dt_val_pr + dt_test_pr + dt_test_roc
    )

    summary = {
        "run_id": RUN_ID,
        "run_tag": args.run_tag,
        "sample_frac": args.sample_frac,
        "shuffle_partitions": args.shuffle_partitions,
        "default_parallelism": args.default_parallelism,
        "val_pr_auc": val_pr,
        "test_pr_auc": test_pr,
        "test_roc_auc": test_roc,
        "test_recall_top5": rec_top5,
        "test_topn": topn,
        "test_n": nrows,
        "time_sec_total": total_time,
        "time_sec_fit": dt_fit,
        "time_sec_pred_test": dt_pred_te
    }

    print("SUMMARY:", summary)

    # 6) Export (JSON + Parquet row)
    safe_write_json_text(spark, summary, OUT + "summary_json_text/")

    spark.createDataFrame([summary]).coalesce(1).write.mode("overwrite").parquet(OUT + "summary_parquet/")
    spark.createDataFrame([summary]).coalesce(1).write.mode("overwrite").csv(OUT + "summary_csv/", header=True)

    print("WROTE:", OUT)

if __name__ == "__main__":
    main()
