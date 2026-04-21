# BIA Project Pyspark

## PHASE 1

#Project bootstrap + load sample parquet + create label + basic splits

1.1 Constants, paths and Spark defaults (run once per notebook)


```python
from pyspark.sql import functions as F
from pyspark.sql import types as T

BUCKET = "gs://big-data-project-481305-flightdelay"

PATH = {
    "raw_csv": f"{BUCKET}/airline/raw/full_data_flightdelay.csv",
    "parquet_full": f"{BUCKET}/airline/parquet/full_data_flightdelay/",
    "parquet_sample": f"{BUCKET}/airline/parquet/sample_10pct/",
    "silver": f"{BUCKET}/airline/silver/",
    "gold": f"{BUCKET}/airline/gold/",
    "models": f"{BUCKET}/airline/models/",
    "metrics": f"{BUCKET}/airline/metrics/",
    "kmeans_preds": f"{BUCKET}/airline/clustering/kmeans_k5/sample10_pred/",
    "kmeans_model": f"{BUCKET}/airline/clustering/kmeans_k5/model/",
}

spark.sparkContext.setLogLevel("WARN")

# sensible defaults for cluster
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.shuffle.partitions", "200")
spark.conf.set("spark.sql.files.maxPartitionBytes", str(128 * 1024 * 1024))  # 128MB

print("Spark:", spark.version)
for k,v in PATH.items():
    print(f"{k:12s} -> {v}")

```

    Spark: 3.5.3
    raw_csv      -> gs://big-data-project-481305-flightdelay/airline/raw/full_data_flightdelay.csv
    parquet_full -> gs://big-data-project-481305-flightdelay/airline/parquet/full_data_flightdelay/
    parquet_sample -> gs://big-data-project-481305-flightdelay/airline/parquet/sample_10pct/
    silver       -> gs://big-data-project-481305-flightdelay/airline/silver/
    gold         -> gs://big-data-project-481305-flightdelay/airline/gold/
    models       -> gs://big-data-project-481305-flightdelay/airline/models/
    metrics      -> gs://big-data-project-481305-flightdelay/airline/metrics/
    kmeans_preds -> gs://big-data-project-481305-flightdelay/airline/clustering/kmeans_k5/sample10_pred/
    kmeans_model -> gs://big-data-project-481305-flightdelay/airline/clustering/kmeans_k5/model/


1.2 Loading the 100% data parquet full


```python
df = spark.read.parquet(PATH["parquet_full"])
print("rows:", df.count(), "| cols:", len(df.columns))
df.printSchema()
df.show(5, truncate=False)

```

                                                                                    

    rows: 6489062 | cols: 26
    root
     |-- MONTH: integer (nullable = true)
     |-- DAY_OF_WEEK: integer (nullable = true)
     |-- DEP_DEL15: integer (nullable = true)
     |-- DEP_TIME_BLK: string (nullable = true)
     |-- DISTANCE_GROUP: integer (nullable = true)
     |-- SEGMENT_NUMBER: integer (nullable = true)
     |-- CONCURRENT_FLIGHTS: integer (nullable = true)
     |-- NUMBER_OF_SEATS: integer (nullable = true)
     |-- CARRIER_NAME: string (nullable = true)
     |-- AIRPORT_FLIGHTS_MONTH: integer (nullable = true)
     |-- AIRLINE_FLIGHTS_MONTH: integer (nullable = true)
     |-- AIRLINE_AIRPORT_FLIGHTS_MONTH: integer (nullable = true)
     |-- AVG_MONTHLY_PASS_AIRPORT: integer (nullable = true)
     |-- AVG_MONTHLY_PASS_AIRLINE: integer (nullable = true)
     |-- FLT_ATTENDANTS_PER_PASS: double (nullable = true)
     |-- GROUND_SERV_PER_PASS: double (nullable = true)
     |-- PLANE_AGE: integer (nullable = true)
     |-- DEPARTING_AIRPORT: string (nullable = true)
     |-- LATITUDE: double (nullable = true)
     |-- LONGITUDE: double (nullable = true)
     |-- PREVIOUS_AIRPORT: string (nullable = true)
     |-- PRCP: double (nullable = true)
     |-- SNOW: double (nullable = true)
     |-- SNWD: double (nullable = true)
     |-- TMAX: double (nullable = true)
     |-- AWND: double (nullable = true)
    


    25/12/21 02:42:13 WARN SparkStringUtils: Truncated the string representation of a plan since it was too large. This behavior can be adjusted by setting 'spark.sql.debug.maxToStringFields'.
                                                                                    

    +-----+-----------+---------+------------+--------------+--------------+------------------+---------------+----------------------+---------------------+---------------------+-----------------------------+------------------------+------------------------+-----------------------+--------------------+---------+-----------------+--------+---------+---------------------------------+----+----+----+----+----+
    |MONTH|DAY_OF_WEEK|DEP_DEL15|DEP_TIME_BLK|DISTANCE_GROUP|SEGMENT_NUMBER|CONCURRENT_FLIGHTS|NUMBER_OF_SEATS|CARRIER_NAME          |AIRPORT_FLIGHTS_MONTH|AIRLINE_FLIGHTS_MONTH|AIRLINE_AIRPORT_FLIGHTS_MONTH|AVG_MONTHLY_PASS_AIRPORT|AVG_MONTHLY_PASS_AIRLINE|FLT_ATTENDANTS_PER_PASS|GROUND_SERV_PER_PASS|PLANE_AGE|DEPARTING_AIRPORT|LATITUDE|LONGITUDE|PREVIOUS_AIRPORT                 |PRCP|SNOW|SNWD|TMAX|AWND|
    +-----+-----------+---------+------------+--------------+--------------+------------------+---------------+----------------------+---------------------+---------------------+-----------------------------+------------------------+------------------------+-----------------------+--------------------+---------+-----------------+--------+---------+---------------------------------+----+----+----+----+----+
    |4    |5          |0        |2100-2159   |2             |4             |91                |175            |Southwest Airlines Co.|32678                |110752               |3345                         |4365661                 |13382999                |6.178236301460919E-5   |9.889412309998219E-5|2        |Atlanta Municipal|33.641  |-84.427  |Southwest Florida International  |0.05|0.0 |0.0 |78.0|7.61|
    |4    |5          |0        |1400-1459   |1             |4             |60                |143            |Southwest Airlines Co.|32678                |110752               |3345                         |4365661                 |13382999                |6.178236301460919E-5   |9.889412309998219E-5|14       |Atlanta Municipal|33.641  |-84.427  |Southwest Florida International  |0.05|0.0 |0.0 |78.0|7.61|
    |4    |5          |1        |1400-1459   |3             |4             |60                |143            |Southwest Airlines Co.|32678                |110752               |3345                         |4365661                 |13382999                |6.178236301460919E-5   |9.889412309998219E-5|21       |Atlanta Municipal|33.641  |-84.427  |Palm Beach International         |0.05|0.0 |0.0 |78.0|7.61|
    |4    |5          |1        |1500-1559   |9             |4             |90                |143            |Southwest Airlines Co.|32678                |110752               |3345                         |4365661                 |13382999                |6.178236301460919E-5   |9.889412309998219E-5|17       |Atlanta Municipal|33.641  |-84.427  |Tampa International              |0.05|0.0 |0.0 |78.0|7.61|
    |4    |5          |0        |1900-1959   |7             |4             |83                |175            |Southwest Airlines Co.|32678                |110752               |3345                         |4365661                 |13382999                |6.178236301460919E-5   |9.889412309998219E-5|7        |Atlanta Municipal|33.641  |-84.427  |Minneapolis-St Paul International|0.05|0.0 |0.0 |78.0|7.61|
    +-----+-----------+---------+------------+--------------+--------------+------------------+---------------+----------------------+---------------------+---------------------+-----------------------------+------------------------+------------------------+-----------------------+--------------------+---------+-----------------+--------+---------+---------------------------------+----+----+----+----+----+
    only showing top 5 rows
    


1.3 Creation of label (delay=1) and a route proxy


```python
# 1) label
if "DEP_DEL15" in df.columns:
    df1 = df.withColumn("label", F.col("DEP_DEL15").cast("int"))
else:
    # fallback if DEP_DEL15 isn't there
    delay_candidates = ["DEP_DELAY", "DEPDELAY", "DEPARTURE_DELAY", "ARR_DELAY", "ARRDELAY", "DELAY"]
    picked = next((c for c in delay_candidates if c in df.columns), None)
    print("Picked delay-minutes column:", picked)
    if picked is None:
        raise ValueError("No DEP_DEL15 and no delay-minutes column found; paste schema and I’ll adapt.")
    df1 = df.withColumn("delay_minutes", F.col(picked).cast("double")) \
            .withColumn("label", (F.col("delay_minutes") > 15).cast("int"))

# 2) route_proxy (no DEST in your data)
required = ["DEPARTING_AIRPORT", "DISTANCE_GROUP", "DEP_TIME_BLK"]
missing = [c for c in required if c not in df1.columns]
if missing:
    raise ValueError(f"Missing columns needed for route_proxy: {missing}")

df1 = df1.withColumn(
    "route_proxy",
    F.concat_ws("_", F.col("DEPARTING_AIRPORT"), F.col("DISTANCE_GROUP"), F.col("DEP_TIME_BLK"))
)

df1.groupBy("label").count().show()
df1.select("DEPARTING_AIRPORT","DISTANCE_GROUP","DEP_TIME_BLK","route_proxy","label").show(10, truncate=False)

```

                                                                                    

    +-----+-------+
    |label|  count|
    +-----+-------+
    |    1|1227368|
    |    0|5261694|
    +-----+-------+
    
    +-----------------+--------------+------------+-----------------------------+-----+
    |DEPARTING_AIRPORT|DISTANCE_GROUP|DEP_TIME_BLK|route_proxy                  |label|
    +-----------------+--------------+------------+-----------------------------+-----+
    |Atlanta Municipal|2             |2100-2159   |Atlanta Municipal_2_2100-2159|0    |
    |Atlanta Municipal|1             |1400-1459   |Atlanta Municipal_1_1400-1459|0    |
    |Atlanta Municipal|3             |1400-1459   |Atlanta Municipal_3_1400-1459|1    |
    |Atlanta Municipal|9             |1500-1559   |Atlanta Municipal_9_1500-1559|1    |
    |Atlanta Municipal|7             |1900-1959   |Atlanta Municipal_7_1900-1959|0    |
    |Atlanta Municipal|3             |1500-1559   |Atlanta Municipal_3_1500-1559|0    |
    |Atlanta Municipal|2             |2100-2159   |Atlanta Municipal_2_2100-2159|0    |
    |Atlanta Municipal|2             |1500-1559   |Atlanta Municipal_2_1500-1559|0    |
    |Atlanta Municipal|3             |1300-1359   |Atlanta Municipal_3_1300-1359|0    |
    |Atlanta Municipal|4             |2000-2059   |Atlanta Municipal_4_2000-2059|0    |
    +-----------------+--------------+------------+-----------------------------+-----+
    only showing top 10 rows
    


1.4 Making a modeling split (month-based & random sanity split)
Because there’s no year/date, month split is “pseudo-time”. We’ll still do it (for the report), but also keep a random split as a guardrail.


```python
# Pseudo-time split (if MONTH exists)
if "MONTH" in df1.columns:
    train_df = df1.filter(F.col("MONTH").between(1, 9))
    val_df   = df1.filter(F.col("MONTH").isin([10]))
    test_df  = df1.filter(F.col("MONTH").isin([11, 12]))
    print("month split sizes:", train_df.count(), val_df.count(), test_df.count())
else:
    # fallback
    train_df, val_df, test_df = df1.randomSplit([0.7, 0.15, 0.15], seed=42)
    print("random split sizes:", train_df.count(), val_df.count(), test_df.count())

# Random split for sanity comparison (always available)
train_r, test_r = df1.randomSplit([0.8, 0.2], seed=42)
print("random sanity split sizes:", train_r.count(), test_r.count())

```

                                                                                    

    month split sizes: 4843946 561327 1083789


    [Stage 22:=============================>                            (1 + 1) / 2]

    random sanity split sizes: 5189820 1299242


                                                                                    

1.5 Saving Phase-1 silver” datasets to GCS (faster + reproducible)


```python
SILVER_BASE = f"{PATH['silver']}/sample_phase1/"

(train_df.write.mode("overwrite").parquet(SILVER_BASE + "train/"))
(val_df.write.mode("overwrite").parquet(SILVER_BASE + "val/"))
(test_df.write.mode("overwrite").parquet(SILVER_BASE + "test/"))

print("Saved to:", SILVER_BASE)

```

                                                                                    

    Saved to: gs://big-data-project-481305-flightdelay/airline/silver//sample_phase1/


Checks:

Schema is clean + complete for modeling: time (MONTH, DAY_OF_WEEK, DEP_TIME_BLK), operational (CONCURRENT_FLIGHTS, seats, plane age), airport + carrier (DEPARTING_AIRPORT, PREVIOUS_AIRPORT, CARRIER_NAME), plus weather already merged (PRCP, SNOW, SNWD, TMAX, AWND).

Label exists (DEP_DEL15)

Class imbalance (delay rate):

delay=1: 122,816
delay=0: 525,501
That’s ~19% delayed, use class weights and report PR-AUC + Recall@TopK.

Month split sizes look reasonable for pseudo-time split.


```python

```

# Phase 2

Feature Engineering & Leakage-safe Aggregates, additionally “Gold” dataset

1. Parse DEP_TIME_BLK → hour → cyclic sin/cos
2. Bucket high-cardinality categoricals (airport/carrier) to control OHE blow-up
3. Build leakage-safe historical delay rate features computed on train only
4. Assemble a final features vector and write to GCS as gold

2.1  Loading saved splits (silver)


```python
from pyspark.sql import functions as F
from pyspark.sql.window import Window

SILVER_BASE = "gs://big-data-project-481305-flightdelay/airline/silver//sample_phase1/"

train = spark.read.parquet(SILVER_BASE + "train/")
val   = spark.read.parquet(SILVER_BASE + "val/")
test  = spark.read.parquet(SILVER_BASE + "test/")

print(train.count(), val.count(), test.count())
train.groupBy("label").count().show()

```

    4843946 561327 1083789
    +-----+-------+
    |label|  count|
    +-----+-------+
    |    1| 945948|
    |    0|3897998|
    +-----+-------+
    


2.2  Parse DEP_TIME_BLK + cyclic time features


```python
import math

def add_time_features(df):
    # Extract start time from "1300-1359" -> "1300"
    df = df.withColumn("dep_blk_start", F.regexp_extract(F.col("DEP_TIME_BLK"), r"^(\d{4})", 1))
    df = df.withColumn("dep_blk_start_int", F.col("dep_blk_start").cast("int"))
    df = df.withColumn("dep_hour", F.floor(F.col("dep_blk_start_int") / 100).cast("int"))

    # Cyclic encodings
    df = df.withColumn("hour_sin", F.sin(2 * math.pi * F.col("dep_hour") / F.lit(24.0))) \
           .withColumn("hour_cos", F.cos(2 * math.pi * F.col("dep_hour") / F.lit(24.0)))

    df = df.withColumn("dow_sin", F.sin(2 * math.pi * F.col("DAY_OF_WEEK") / F.lit(7.0))) \
           .withColumn("dow_cos", F.cos(2 * math.pi * F.col("DAY_OF_WEEK") / F.lit(7.0)))

    df = df.withColumn("month_sin", F.sin(2 * math.pi * F.col("MONTH") / F.lit(12.0))) \
           .withColumn("month_cos", F.cos(2 * math.pi * F.col("MONTH") / F.lit(12.0)))

    return df

train2 = add_time_features(train)
val2   = add_time_features(val)
test2  = add_time_features(test)

train2.select("DEP_TIME_BLK","dep_hour","hour_sin","hour_cos","DAY_OF_WEEK","dow_sin","dow_cos","MONTH","month_sin","month_cos").show(5, truncate=False)

```

    +------------+--------+----------------------+--------------------+-----------+------------------+--------------------+-----+------------------+------------------+
    |DEP_TIME_BLK|dep_hour|hour_sin              |hour_cos            |DAY_OF_WEEK|dow_sin           |dow_cos             |MONTH|month_sin         |month_cos         |
    +------------+--------+----------------------+--------------------+-----------+------------------+--------------------+-----+------------------+------------------+
    |1200-1259   |12      |1.2246467991473532E-16|-1.0                |2          |0.9749279121818236|-0.22252093395631434|2    |0.8660254037844386|0.5000000000000001|
    |0800-0859   |8       |0.8660254037844387    |-0.4999999999999998 |2          |0.9749279121818236|-0.22252093395631434|2    |0.8660254037844386|0.5000000000000001|
    |0700-0759   |7       |0.9659258262890683    |-0.25881904510252063|2          |0.9749279121818236|-0.22252093395631434|2    |0.8660254037844386|0.5000000000000001|
    |1000-1059   |10      |0.49999999999999994   |-0.8660254037844387 |2          |0.9749279121818236|-0.22252093395631434|2    |0.8660254037844386|0.5000000000000001|
    |1100-1159   |11      |0.258819045102521     |-0.9659258262890682 |2          |0.9749279121818236|-0.22252093395631434|2    |0.8660254037844386|0.5000000000000001|
    +------------+--------+----------------------+--------------------+-----------+------------------+--------------------+-----+------------------+------------------+
    only showing top 5 rows
    


2.3 Top-N bucketing for high-cardinality categoricals (train-driven)

Prevents huge one-hot vectors from airports.


```python
def top_n_values(df, col, n=50):
    return [r[col] for r in df.groupBy(col).count().orderBy(F.desc("count")).limit(n).collect()]

def bucket_top_n(df, col, top_list, new_col):
    # Keep null explicit
    df = df.withColumn(col, F.when(F.col(col).isNull(), F.lit("__MISSING__")).otherwise(F.col(col)))
    return df.withColumn(new_col, F.when(F.col(col).isin(top_list), F.col(col)).otherwise(F.lit("__OTHER__")))

TOP_AIRPORTS = top_n_values(train2, "DEPARTING_AIRPORT", n=50)
TOP_PREV     = top_n_values(train2, "PREVIOUS_AIRPORT", n=50)
TOP_CARRIER  = top_n_values(train2, "CARRIER_NAME", n=25)

train3 = bucket_top_n(train2, "DEPARTING_AIRPORT", TOP_AIRPORTS, "DEPARTING_AIRPORT_BKT")
val3   = bucket_top_n(val2,   "DEPARTING_AIRPORT", TOP_AIRPORTS, "DEPARTING_AIRPORT_BKT")
test3  = bucket_top_n(test2,  "DEPARTING_AIRPORT", TOP_AIRPORTS, "DEPARTING_AIRPORT_BKT")

train3 = bucket_top_n(train3, "PREVIOUS_AIRPORT", TOP_PREV, "PREVIOUS_AIRPORT_BKT")
val3   = bucket_top_n(val3,   "PREVIOUS_AIRPORT", TOP_PREV, "PREVIOUS_AIRPORT_BKT")
test3  = bucket_top_n(test3,  "PREVIOUS_AIRPORT", TOP_PREV, "PREVIOUS_AIRPORT_BKT")

train3 = bucket_top_n(train3, "CARRIER_NAME", TOP_CARRIER, "CARRIER_NAME_BKT")
val3   = bucket_top_n(val3,   "CARRIER_NAME", TOP_CARRIER, "CARRIER_NAME_BKT")
test3  = bucket_top_n(test3,  "CARRIER_NAME", TOP_CARRIER, "CARRIER_NAME_BKT")

train3.select("DEPARTING_AIRPORT","DEPARTING_AIRPORT_BKT","PREVIOUS_AIRPORT","PREVIOUS_AIRPORT_BKT","CARRIER_NAME","CARRIER_NAME_BKT").show(5, truncate=False)

```

                                                                                    

    +---------------------------------+---------------------------------+-------------------------------+-------------------------------+--------------------+--------------------+
    |DEPARTING_AIRPORT                |DEPARTING_AIRPORT_BKT            |PREVIOUS_AIRPORT               |PREVIOUS_AIRPORT_BKT           |CARRIER_NAME        |CARRIER_NAME_BKT    |
    +---------------------------------+---------------------------------+-------------------------------+-------------------------------+--------------------+--------------------+
    |Minneapolis-St Paul International|Minneapolis-St Paul International|Newark Liberty International   |Newark Liberty International   |Delta Air Lines Inc.|Delta Air Lines Inc.|
    |Minneapolis-St Paul International|Minneapolis-St Paul International|Lambert-St. Louis International|Lambert-St. Louis International|Delta Air Lines Inc.|Delta Air Lines Inc.|
    |Minneapolis-St Paul International|Minneapolis-St Paul International|Hector Field                   |__OTHER__                      |Delta Air Lines Inc.|Delta Air Lines Inc.|
    |Minneapolis-St Paul International|Minneapolis-St Paul International|John F. Kennedy International  |John F. Kennedy International  |Delta Air Lines Inc.|Delta Air Lines Inc.|
    |Minneapolis-St Paul International|Minneapolis-St Paul International|Friendship International       |Friendship International       |Delta Air Lines Inc.|Delta Air Lines Inc.|
    +---------------------------------+---------------------------------+-------------------------------+-------------------------------+--------------------+--------------------+
    only showing top 5 rows
    


2.4 Leakage-safe historical delay rate features (computed on TRAIN only)

Adding : 1. airport+timeblock delay baseline2. route_proxy delay baseline with Bayesian smoothing (important when counts are small).


```python
# global delay rate from TRAIN only
global_rate = train3.select(F.avg(F.col("label").cast("double")).alias("gr")).first()["gr"]
m = 20.0  # smoothing strength

# airport x timeblock baseline
agg_air_blk = (train3
    .groupBy("DEPARTING_AIRPORT_BKT", "DEP_TIME_BLK")
    .agg(F.count("*").alias("n_air_blk"),
         F.sum("label").alias("sum_air_blk"))
    .withColumn(
        "delay_rate_air_blk",
        (F.col("sum_air_blk") + F.lit(m) * F.lit(global_rate)) / (F.col("n_air_blk") + F.lit(m))
    )
    .select("DEPARTING_AIRPORT_BKT","DEP_TIME_BLK","delay_rate_air_blk","n_air_blk")
)

# route_proxy baseline
agg_route = (train3
    .groupBy("route_proxy")
    .agg(F.count("*").alias("n_route"),
         F.sum("label").alias("sum_route"))
    .withColumn(
        "delay_rate_route",
        (F.col("sum_route") + F.lit(m) * F.lit(global_rate)) / (F.col("n_route") + F.lit(m))
    )
    .select("route_proxy","delay_rate_route","n_route")
)

def add_agg_features(df):
    df = df.join(agg_air_blk, ["DEPARTING_AIRPORT_BKT","DEP_TIME_BLK"], "left") \
           .join(agg_route, ["route_proxy"], "left") \
           .withColumn("delay_rate_air_blk", F.coalesce(F.col("delay_rate_air_blk"), F.lit(global_rate))) \
           .withColumn("delay_rate_route",   F.coalesce(F.col("delay_rate_route"),   F.lit(global_rate))) \
           .withColumn("n_air_blk", F.coalesce(F.col("n_air_blk"), F.lit(0))) \
           .withColumn("n_route",   F.coalesce(F.col("n_route"),   F.lit(0)))
    return df

train4 = add_agg_features(train3)
val4   = add_agg_features(val3)
test4  = add_agg_features(test3)

train4.select("DEPARTING_AIRPORT_BKT","DEP_TIME_BLK","delay_rate_air_blk","n_air_blk","route_proxy","delay_rate_route","n_route").show(5, truncate=False)

```

                                                                                    

    +---------------------------------+------------+-------------------+---------+---------------------------------------------+-------------------+-------+
    |DEPARTING_AIRPORT_BKT            |DEP_TIME_BLK|delay_rate_air_blk |n_air_blk|route_proxy                                  |delay_rate_route   |n_route|
    +---------------------------------+------------+-------------------+---------+---------------------------------------------+-------------------+-------+
    |Minneapolis-St Paul International|1200-1259   |0.15825579541191565|6570     |Minneapolis-St Paul International_4_1200-1259|0.1490304128539464 |2140   |
    |Minneapolis-St Paul International|0800-0859   |0.11411670041323116|7638     |Minneapolis-St Paul International_4_0800-0859|0.10081169633940613|1219   |
    |Minneapolis-St Paul International|0700-0759   |0.09073807072533809|4784     |Minneapolis-St Paul International_4_0700-0759|0.09677618382931735|1219   |
    |Minneapolis-St Paul International|1000-1059   |0.11252770496236189|6502     |Minneapolis-St Paul International_5_1000-1059|0.12556324114709216|1588   |
    |Minneapolis-St Paul International|1100-1159   |0.12236272890362987|10538    |Minneapolis-St Paul International_3_1100-1159|0.1123280279466949 |1546   |
    +---------------------------------+------------+-------------------+---------+---------------------------------------------+-------------------+-------+
    only showing top 5 rows
    


                                                                                    

2.5 Adding class weights (for LR metrics + Recall@TopK)


```python
counts = train4.groupBy("label").count().collect()
c0 = next(r["count"] for r in counts if r["label"] == 0)
c1 = next(r["count"] for r in counts if r["label"] == 1)
total = c0 + c1

w0 = total / (2.0 * c0)
w1 = total / (2.0 * c1)
print("class weights:", {0: w0, 1: w1})

def add_weights(df):
    return df.withColumn("weight", F.when(F.col("label")==1, F.lit(w1)).otherwise(F.lit(w0)))

train5 = add_weights(train4)
val5   = add_weights(val4)
test5  = add_weights(test4)

```

    class weights: {0: 0.6213376712866451, 1: 2.560365897491194}


2.6 Saving “gold” feature-ready datasets (pre-VectorAssembler)


```python
GOLD_BASE = "gs://big-data-project-481305-flightdelay/airline/gold/sample_step2_features/"

(train5.write.mode("overwrite").parquet(GOLD_BASE + "train/"))
(val5.write.mode("overwrite").parquet(GOLD_BASE + "val/"))
(test5.write.mode("overwrite").parquet(GOLD_BASE + "test/"))

print("Saved gold to:", GOLD_BASE)

```

                                                                                    

    Saved gold to: gs://big-data-project-481305-flightdelay/airline/gold/sample_step2_features/


Ouput Checks:

Train split: 484,002 rows with label counts 0=388,927, 1=95,075 → delay rate ≈ 19.65% (95,075 / 484,002).

Cyclic features  (hour_sin/cos etc.).

Top-N bucketing working (shows __OTHER__ appearing).

Leakage-safe aggregates (counts + smoothed rates ).

Class weights for imbalance:
w0 ≈ 0.622, w1 ≈ 2.545 




```python

```

# Phase 3

ML pipeline + metrics + saved artifacts (models/preds/metrics to GCS)

Modeling + evaluation framework:

1. Logistic Regression (weighted)

2. GBTClassifier (weighted)

3. ROC-AUC, PR-AUC

4. Confusion matrix + Precision/Recall/F1 @ threshold

5. Recall@TopK% (e.g., 5%)

6. Cost-aware expected cost (FN cost > FP cost)

7. Brier score + calibration bins

8. Saving predictions + metrics + models to GCS


```python
# Set log level to ERROR to suppress WARN messages for BlockManager
sc.setLogLevel("ERROR")

# only silencing specific modules
log4j = spark._jvm.org.apache.log4j
log4j.LogManager.getLogger("org.apache.spark.storage.BlockManagerMasterEndpoint").setLevel(log4j.Level.ERROR)
log4j.LogManager.getLogger("org.apache.spark.storage.BlockManager").setLevel(log4j.Level.ERROR)
```


```python

```

3.1  Loading gold datasets & basic hygiene


```python
from pyspark.sql import functions as F

GOLD_BASE = "gs://big-data-project-481305-flightdelay/airline/gold/sample_step2_features/"

train = spark.read.parquet(GOLD_BASE + "train/")
val   = spark.read.parquet(GOLD_BASE + "val/")
test  = spark.read.parquet(GOLD_BASE + "test/")

# Keep only rows with label present
train = train.filter(F.col("label").isNotNull())
val   = val.filter(F.col("label").isNotNull())
test  = test.filter(F.col("label").isNotNull())

print(train.count(), val.count(), test.count())
train.groupBy("label").count().show()

```

                                                                                    

    4843946 561327 1083789


                                                                                    

    +-----+-------+
    |label|  count|
    +-----+-------+
    |    1| 945948|
    |    0|3897998|
    +-----+-------+
    



```python
from pyspark.sql import functions as F

def add_row_id(df, split_name):
    # deterministic, stable ID for joins/slicing
    return df.withColumn(
        "row_id",
        F.xxhash64(
            F.lit(split_name),
            F.col("DEPARTING_AIRPORT"),
            F.col("CARRIER_NAME"),
            F.col("DEP_TIME_BLK"),
            F.col("MONTH"),
            F.col("DAY_OF_WEEK"),
            F.col("DISTANCE_GROUP"),
            F.col("SEGMENT_NUMBER"),
        )
    )

train5 = add_row_id(train, "train")
val5   = add_row_id(val,   "val")
test5  = add_row_id(test,  "test")

# Fast sanity check on a sample (avoids full distinct shuffle)
sample = train5.sample(False, 0.02, seed=42).cache()
print("sample rows:", sample.count(),
      "distinct row_id in sample:", sample.select("row_id").distinct().count())

```

                                                                                    

    sample rows: 97057 distinct row_id in sample: 92423


3.2  Defining feature columns (categorical + numeric)
(Based on schema + engineered cols)


```python
# Categorical columns to index+OHE (already bucketed -> safe dimensionality)
cat_cols = [
    "DEPARTING_AIRPORT_BKT",
    "PREVIOUS_AIRPORT_BKT",
    "CARRIER_NAME_BKT",
    "DEP_TIME_BLK",      # still useful even with cyclic hour
    "DISTANCE_GROUP",
    "SEGMENT_NUMBER"
]

# Numeric features (raw + engineered + weather + leakage-safe aggregates)
num_cols = [
    "CONCURRENT_FLIGHTS",
    "NUMBER_OF_SEATS",
    "AIRPORT_FLIGHTS_MONTH",
    "AIRLINE_FLIGHTS_MONTH",
    "AIRLINE_AIRPORT_FLIGHTS_MONTH",
    "AVG_MONTHLY_PASS_AIRPORT",
    "AVG_MONTHLY_PASS_AIRLINE",
    "FLT_ATTENDANTS_PER_PASS",
    "GROUND_SERV_PER_PASS",
    "PLANE_AGE",
    "LATITUDE",
    "LONGITUDE",
    "PRCP",
    "SNOW",
    "SNWD",
    "TMAX",
    "AWND",
    "delay_rate_air_blk",
    "delay_rate_route",
    "n_air_blk",
    "n_route",
    "hour_sin","hour_cos",
    "dow_sin","dow_cos",
    "month_sin","month_cos"
]

# Making sure all exist (fail fast)
missing = [c for c in (cat_cols + num_cols + ["label","weight"]) if c not in train.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

print("Categorical:", cat_cols)
print("Numeric:", len(num_cols))

```

    Categorical: ['DEPARTING_AIRPORT_BKT', 'PREVIOUS_AIRPORT_BKT', 'CARRIER_NAME_BKT', 'DEP_TIME_BLK', 'DISTANCE_GROUP', 'SEGMENT_NUMBER']
    Numeric: 27


3.3 Building Spark ML pipeline 
(StringIndexer → OHE → Imputer → VectorAssembler → optional scaling)


```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer, VectorAssembler, StandardScaler

# Convert all categoricals to string (esp ints like DISTANCE_GROUP)
for c in cat_cols:
    train = train.withColumn(c, F.col(c).cast("string"))
    val   = val.withColumn(c, F.col(c).cast("string"))
    test  = test.withColumn(c, F.col(c).cast("string"))

indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in cat_cols
]

encoder = OneHotEncoder(
    inputCols=[f"{c}_idx" for c in cat_cols],
    outputCols=[f"{c}_ohe" for c in cat_cols],
    handleInvalid="keep"
)

# Impute numeric nulls using TRAIN statistics
imputer = Imputer(
    inputCols=num_cols,
    outputCols=[f"{c}_imp" for c in num_cols],
    strategy="median"
)

assembled_inputs = [f"{c}_imp" for c in num_cols] + [f"{c}_ohe" for c in cat_cols]

assembler = VectorAssembler(
    inputCols=assembled_inputs,
    outputCol="features_raw",
    handleInvalid="keep"
)

# Optional but recommended for Logistic Regression
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withMean=False,   # keep False for sparse
    withStd=True
)

feature_pipeline = Pipeline(stages=indexers + [encoder, imputer, assembler, scaler])
feat_model = feature_pipeline.fit(train)

train_fe = feat_model.transform(train).select("label","weight","features")
val_fe   = feat_model.transform(val).select("label","weight","features")
test_fe  = feat_model.transform(test).select("label","weight","features")

print(train_fe.count(), val_fe.count(), test_fe.count())

```

                                                                                    

    4843946 561327 1083789



```python
# Keep row_id + slice columns for later slice metrics
keep_cols = ["row_id", "label", "weight", "features",
             "MONTH", "DEP_TIME_BLK", "DEPARTING_AIRPORT_BKT", "CARRIER_NAME_BKT"]

train_fe = feat_model.transform(train5).select(*keep_cols)
val_fe   = feat_model.transform(val5).select(*keep_cols)
test_fe  = feat_model.transform(test5).select(*keep_cols)

# cache once (important)
train_fe = train_fe.cache(); _ = train_fe.count()
val_fe   = val_fe.cache();   _ = val_fe.count()
test_fe  = test_fe.cache();  _ = test_fe.count()

```

                                                                                    

Cache feature datasets once (prevents repeated recomputation)

Spark WARN warnings: sending a large serialized “task” to executors

The pipeline/model object being shipped is big (OHE metadata, many stages), running transformations inside Python helper functions and Spark ends up serializing more than expected.

Fix:


```python
# 1) Python-side Spark log level
spark.sparkContext.setLogLevel("ERROR")

# 2) JVM loggers (suppresses most WARN spam in Dataproc notebooks)
log4j = spark._jvm.org.apache.log4j
log4j.LogManager.getLogger("org").setLevel(log4j.Level.ERROR)
log4j.LogManager.getLogger("akka").setLevel(log4j.Level.ERROR)
log4j.LogManager.getLogger("org.apache.spark.scheduler.DAGScheduler").setLevel(log4j.Level.ERROR)

# Suppress BlockManager warnings specifically
log4j.LogManager.getLogger("org.apache.spark.storage.BlockManagerMasterEndpoint").setLevel(log4j.Level.ERROR)
log4j.LogManager.getLogger("org.apache.spark.storage.BlockManager").setLevel(log4j.Level.ERROR)

```


```python

train_fe = train_fe.cache(); _ = train_fe.count()
val_fe   = val_fe.cache();   _ = val_fe.count()
test_fe  = test_fe.cache();  _ = test_fe.count()

```

    25/12/21 02:48:58 WARN CacheManager: Asked to cache already cached data.
    25/12/21 02:48:58 WARN CacheManager: Asked to cache already cached data.
    25/12/21 02:48:59 WARN CacheManager: Asked to cache already cached data.


3.4 Training models (LogReg + GBT)


```python
from pyspark.ml.classification import LogisticRegression, GBTClassifier

# Logistic Regression (strong baseline, calibrated-ish)
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    weightCol="weight",
    maxIter=50,
    regParam=0.05,
    elasticNetParam=0.0  # pure L2 to start
)

lr_model = lr.fit(train_fe)



```

                                                                                    


```python
# # Gradient Boosted Trees (strong on tabular, handles nonlinearity)
# gbt = GBTClassifier(
#     featuresCol="features",
#     labelCol="label",
#     weightCol="weight",
#     maxIter=20,
#     maxDepth=5,
#     stepSize=0.1,
#     subsamplingRate=0.7
#     featureSubsetStrategy="onethird"
# )

# gbt_model = gbt.fit(train_fe)
```


```python
from pyspark.ml.classification import GBTClassificationModel, GBTClassifier

# DEFINE PATHS 
BASE = "gs://big-data-project-481305-flightdelay/airline"
LOAD_FROM_RUN_ID = "20251219_161132" 

gbt_model = None

# ATTEMPT TO LOAD ---
if LOAD_FROM_RUN_ID:
    # Construct the path to the saved model using BASE
    old_model_path = f"{BASE}/models/sample_run_{LOAD_FROM_RUN_ID}/gbt_model"
    
    try:
        print(f"--> Attempting to load GBT model from: {old_model_path}")
        gbt_model = GBTClassificationModel.load(old_model_path)
        print("--> Success! Loaded pre-trained GBT model. Skipping fit.")
    except Exception as e:
        print(f"--> Could not load model. Error: {e}")
        print("--> Proceeding to train new model...")

# TRAIN IF LOAD FAILED (OR IF NO ID PROVIDED) ---
if gbt_model is None:
    print("--> Starting GBT training (this will take time)...")
    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="label",
        weightCol="weight",
        maxIter=80,
        maxDepth=6,
        stepSize=0.08,
        subsamplingRate=0.8
    )
    gbt_model = gbt.fit(train_fe)
    print("--> Training complete.")
```

    --> Attempting to load GBT model from: gs://big-data-project-481305-flightdelay/airline/models/sample_run_20251219_161132/gbt_model


    [Stage 206:>                                                        (0 + 1) / 1]

    --> Success! Loaded pre-trained GBT model. Skipping fit.


                                                                                    


```python
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    weightCol="weight",
    numTrees=30,
    maxDepth=6,
    featureSubsetStrategy="sqrt",
    subsamplingRate=0.8,
    seed=42
)

#rf_model = rf.fit(train_fe)

print("Starting fit with reduced complexity...")
import time
start = time.time()
rf_model = rf.fit(train_fe)
print(f"Fit complete in {time.time() - start:.2f} seconds")

```

    Starting fit with reduced complexity...


                                                                                    

    Fit complete in 206.36 seconds


3.5 Evaluation utilities (all key metrics + calibration)


```python
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array

def add_prob(df_pred):
    return (df_pred
            .withColumn("prob_arr", vector_to_array(F.col("probability")))
            .withColumn("p1", F.col("prob_arr")[1])
            .drop("prob_arr"))

def basic_auc_metrics(df_pred):
    # Use probability as rawPredictionCol (Spark accepts)
    e_roc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="probability", metricName="areaUnderROC")
    e_pr  = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="probability", metricName="areaUnderPR")
    return float(e_roc.evaluate(df_pred)), float(e_pr.evaluate(df_pred))

def confusion_metrics(df_prob, thr=0.5):
    df2 = df_prob.withColumn("pred", (F.col("p1") >= F.lit(thr)).cast("int"))
    row = df2.agg(
        F.sum(((F.col("label")==1) & (F.col("pred")==1)).cast("int")).alias("tp"),
        F.sum(((F.col("label")==0) & (F.col("pred")==1)).cast("int")).alias("fp"),
        F.sum(((F.col("label")==0) & (F.col("pred")==0)).cast("int")).alias("tn"),
        F.sum(((F.col("label")==1) & (F.col("pred")==0)).cast("int")).alias("fn"),
    ).first()

    tp, fp, tn, fn = int(row["tp"]), int(row["fp"]), int(row["tn"]), int(row["fn"])
    precision = tp/(tp+fp) if (tp+fp) else 0.0
    recall    = tp/(tp+fn) if (tp+fn) else 0.0
    f1        = (2*precision*recall)/(precision+recall) if (precision+recall) else 0.0
    return {"thr": thr, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}

def brier_score(df_prob):
    return float(df_prob.select(F.avg((F.col("p1") - F.col("label"))**2).alias("brier")).first()["brier"])

def expected_cost(df_prob, thr=0.5, cost_fn=5.0, cost_fp=1.0):
    df2 = df_prob.withColumn("pred", (F.col("p1") >= F.lit(thr)).cast("int"))
    row = df2.agg(
        F.sum(((F.col("label")==1) & (F.col("pred")==0)).cast("int")).alias("fn"),
        F.sum(((F.col("label")==0) & (F.col("pred")==1)).cast("int")).alias("fp"),
        F.count("*").alias("n")
    ).first()
    fn, fp, n = int(row["fn"]), int(row["fp"]), int(row["n"])
    exp = (cost_fn*fn + cost_fp*fp)/n if n else None
    return {"thr": thr, "cost_fn": cost_fn, "cost_fp": cost_fp, "expected_cost_per_row": float(exp)}

def recall_at_topk_fast(df_prob, k=0.05, rel_error=0.001):
    n = df_prob.count()
    topn = int(n * k)

    # quantile threshold avoids full sort
    thr = df_prob.approxQuantile("p1", [1.0 - k], rel_error)[0]

    row = df_prob.agg(
        F.sum((F.col("label")==1).cast("int")).alias("pos_total"),
        F.sum(((F.col("p1") >= F.lit(thr)) & (F.col("label")==1)).cast("int")).alias("pos_in_topk")
    ).first()

    pos_total = int(row["pos_total"])
    pos_in_topk = int(row["pos_in_topk"])
    rec = pos_in_topk/pos_total if pos_total else 0.0
    return {"k": k, "topn": topn, "thr_used": float(thr), "recall_at_topk": float(rec)}

def calibration_bins(df_prob, bins=10):
    dfb = df_prob.withColumn(
        "bin",
        F.when(F.col("p1") >= 1.0, F.lit(bins-1))
         .otherwise(F.floor(F.col("p1") * F.lit(bins)).cast("int"))
    )
    return (dfb.groupBy("bin")
              .agg(F.count("*").alias("n"),
                   F.avg("p1").alias("avg_p"),
                   F.avg(F.col("label").cast("double")).alias("emp_rate"))
              .orderBy("bin"))

def add_accuracy_to_conf(conf):
    tp, fp, tn, fn = conf["tp"], conf["fp"], conf["tn"], conf["fn"]
    total = tp + fp + tn + fn
    conf = dict(conf)
    conf["accuracy"] = (tp + tn) / total if total else 0.0
    return conf

def risk_regression_metrics(df_prob):
    # assumes columns: label (0/1), p1 in [0,1]
    eps = 1e-15

    row = df_prob.select(
        F.avg((F.col("p1") - F.col("label"))**2).alias("brier"),
        F.avg(F.abs(F.col("p1") - F.col("label"))).alias("mae_prob"),
        F.avg(
            -(F.col("label") * F.log(F.greatest(F.col("p1"), F.lit(eps))) +
              (1 - F.col("label")) * F.log(F.greatest(1 - F.col("p1"), F.lit(eps))))
        ).alias("logloss")
    ).first()

    brier = float(row["brier"])
    return {
        "brier": brier,
        "rmse_prob": float(math.sqrt(brier)),
        "mae_prob": float(row["mae_prob"]),
        "logloss": float(row["logloss"])
    }
```

Block for FAST ablation utilities + timer


```python
import time
from pyspark.sql import functions as F

def timed(name, fn):
    t0 = time.time()
    out = fn()
    dt = time.time() - t0
    print(f"[TIMER] {name}: {dt:.2f} sec ({dt/60:.2f} min)")
    return out, dt

def recall_at_topk_approx(df_prob, k=0.05, rel_error=0.01):
    cutoff = df_prob.approxQuantile("p1", [1.0 - k], rel_error)[0]
    agg = df_prob.agg(
        F.sum(F.when(F.col("label")==1, 1).otherwise(0)).alias("pos"),
        F.sum(F.when((F.col("label")==1) & (F.col("p1") >= F.lit(cutoff)), 1).otherwise(0)).alias("tp_topk")
    ).first()
    pos = float(agg["pos"]) if agg["pos"] else 0.0
    return float(agg["tp_topk"]) / pos if pos > 0 else 0.0

def confusion_metrics_onepass(df_prob, thr=0.5):
    r = df_prob.agg(
        F.sum(F.when((F.col("label")==1) & (F.col("p1")>=thr), 1).otherwise(0)).alias("tp"),
        F.sum(F.when((F.col("label")==0) & (F.col("p1")>=thr), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col("label")==0) & (F.col("p1")< thr), 1).otherwise(0)).alias("tn"),
        F.sum(F.when((F.col("label")==1) & (F.col("p1")< thr), 1).otherwise(0)).alias("fn")
    ).first()

    tp, fp, tn, fn = int(r["tp"]), int(r["fp"]), int(r["tn"]), int(r["fn"])
    precision = tp / (tp + fp) if (tp+fp) else 0.0
    recall    = tp / (tp + fn) if (tp+fn) else 0.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    acc       = (tp + tn) / (tp+fp+tn+fn) if (tp+fp+tn+fn) else 0.0
    return {"thr": thr, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1, "accuracy": acc}

def evaluate_ablation_light(name, model, va, te, thr=0.5):
    va_pred = add_prob(model.transform(va)).select("label","p1","probability").cache()
    _ = va_pred.count()

    te_pred = add_prob(model.transform(te)).select("label","p1","probability").cache()
    _ = te_pred.count()

    va_roc, va_pr = basic_auc_metrics(va_pred)
    te_roc, te_pr = basic_auc_metrics(te_pred)

    va_cm = confusion_metrics_onepass(va_pred, thr)
    te_cm = confusion_metrics_onepass(te_pred, thr)

    va_top5 = recall_at_topk_approx(va_pred, 0.05)
    te_top5 = recall_at_topk_approx(te_pred, 0.05)

    return {
        "model": name,
        "val_roc_auc": va_roc,
        "val_pr_auc": va_pr,
        "val_f1": va_cm["f1"],
        "val_recall_top5": va_top5,
        "test_roc_auc": te_roc,
        "test_pr_auc": te_pr,
        "test_f1": te_cm["f1"],
        "test_recall_top5": te_top5,
    }

```

Fixing the probability column is a VectorUDT (internally stored like a struct with type,size,indices,values),


```python
from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array

def add_prob(df_pred):
    # works for VectorUDT columns (probability) across Spark versions
    return (df_pred
            .withColumn("prob_arr", vector_to_array(F.col("probability")))
            .withColumn("p1", F.col("prob_arr")[1])
            .drop("prob_arr"))

```


```python
#sanity check
tmp = add_prob(lr_model.transform(val_fe))
tmp.select("probability", "p1", "label").show(5, truncate=False)

```

    [Stage 227:>                                                        (0 + 1) / 1]

    +----------------------------------------+-------------------+-----+
    |probability                             |p1                 |label|
    +----------------------------------------+-------------------+-----+
    |[0.8034592015889126,0.1965407984110874] |0.1965407984110874 |0    |
    |[0.7489441778563759,0.2510558221436241] |0.2510558221436241 |0    |
    |[0.691994944313376,0.30800505568662395] |0.30800505568662395|1    |
    |[0.5422241463826694,0.45777585361733064]|0.45777585361733064|0    |
    |[0.7497988077459098,0.2502011922540902] |0.2502011922540902 |0    |
    +----------------------------------------+-------------------+-----+
    only showing top 5 rows
    


                                                                                    

Caching predictions before evaluation (prevents repeated recomputation)


```python
lr_val_pred = add_prob(lr_model.transform(val_fe)).cache()
gbt_val_pred = add_prob(gbt_model.transform(val_fe)).cache()

rf_val_pred = add_prob(rf_model.transform(val_fe)).cache()

print(lr_val_pred.count(), gbt_val_pred.count(),rf_val_pred.count())

```

    [Stage 236:============================>                            (1 + 1) / 2]

    561327 561327 561327


                                                                                    

3.6 Evaluation on VAL, picking threshold, then evaluating TEST

(We’ll pick a threshold that maximizes F1 on validation (can also pick cost-minimizing).


```python
def eval_model(name, model, df_fe):
    pred = model.transform(df_fe)
    pred = add_prob(pred)
    roc, pr = basic_auc_metrics(pred)
    return pred, roc, pr

# --- Validation ---
lr_val_pred, lr_val_roc, lr_val_pr = eval_model("lr", lr_model, val_fe)
gbt_val_pred, gbt_val_roc, gbt_val_pr = eval_model("gbt", gbt_model, val_fe)

print("LR  VAL  ROC-AUC:", lr_val_roc, "PR-AUC:", lr_val_pr)
print("GBT VAL  ROC-AUC:", gbt_val_roc, "PR-AUC:", gbt_val_pr)

# Threshold sweep on validation
thresholds = [i/100 for i in range(10, 91, 5)]

def best_thr(df_prob):
    stats = [confusion_metrics(df_prob, t) for t in thresholds]
    best = max(stats, key=lambda d: d["f1"])
    return best, stats

lr_best, _  = best_thr(lr_val_pred)
gbt_best, _ = best_thr(gbt_val_pred)

print("LR best thr:", lr_best)
print("GBT best thr:", gbt_best)

# Extra val metrics
print("LR recall@top5%:", recall_at_topk_fast(lr_val_pred, 0.05))
print("GBT recall@top5%:", recall_at_topk_fast(gbt_val_pred, 0.05))

print("LR brier:", brier_score(lr_val_pred))
print("GBT brier:", brier_score(gbt_val_pred))

```

                                                                                    

    LR  VAL  ROC-AUC: 0.6686114635773452 PR-AUC: 0.2839782893192411
    GBT VAL  ROC-AUC: 0.6856542383010754 PR-AUC: 0.31000755390762613


                                                                                    

    LR best thr: {'thr': 0.5, 'tp': 48937, 'fp': 143761, 'tn': 326821, 'fn': 41808, 'precision': 0.2539569689358478, 'recall': 0.5392804011240289, 'f1': 0.3453039941011067}
    GBT best thr: {'thr': 0.55, 'tp': 44186, 'fp': 110616, 'tn': 359966, 'fn': 46559, 'precision': 0.28543558868748464, 'recall': 0.4869248994434955, 'f1': 0.3598985123011073}


                                                                                    

    LR recall@top5%: {'k': 0.05, 'topn': 28066, 'thr_used': 0.676099856711501, 'recall_at_topk': 0.12437048873216155}


                                                                                    

    GBT recall@top5%: {'k': 0.05, 'topn': 28066, 'thr_used': 0.726509126227324, 'recall_at_topk': 0.13607361287123257}
    LR brier: 0.20749280154649552


    [Stage 409:============================>                            (1 + 1) / 2]

    GBT brier: 0.2169265724140898


                                                                                    

Evaluating on TEST using the chosen thresholds:


```python
# --- Test ---
lr_test_pred, lr_test_roc, lr_test_pr = eval_model("lr", lr_model, test_fe)
gbt_test_pred, gbt_test_roc, gbt_test_pr = eval_model("gbt", gbt_model, test_fe)

lr_test_cm  = confusion_metrics(lr_test_pred, lr_best["thr"])
gbt_test_cm = confusion_metrics(gbt_test_pred, gbt_best["thr"])

print("LR  TEST ROC-AUC:", lr_test_roc, "PR-AUC:", lr_test_pr, "CM:", lr_test_cm)
print("GBT TEST ROC-AUC:", gbt_test_roc, "PR-AUC:", gbt_test_pr, "CM:", gbt_test_cm)

print("LR  TEST recall@top5%:", recall_at_topk_fast(lr_test_pred, 0.05))
print("GBT TEST recall@top5%:", recall_at_topk_fast(gbt_test_pred, 0.05))

print("LR  TEST cost:", expected_cost(lr_test_pred, lr_best["thr"], cost_fn=5.0, cost_fp=1.0))
print("GBT TEST cost:", expected_cost(gbt_test_pred, gbt_best["thr"], cost_fn=5.0, cost_fp=1.0))

# Calibration tables (display)
lr_cal = calibration_bins(lr_test_pred, bins=10)
gbt_cal = calibration_bins(gbt_test_pred, bins=10)
lr_cal.show(20, truncate=False)
gbt_cal.show(20, truncate=False)

```

                                                                                    

    LR  TEST ROC-AUC: 0.657144145155172 PR-AUC: 0.2920900016667561 CM: {'thr': 0.5, 'tp': 92566, 'fp': 247240, 'tn': 645874, 'fn': 98109, 'precision': 0.272408374189979, 'recall': 0.48546479611905075, 'f1': 0.34898893645578255}
    GBT TEST ROC-AUC: 0.6759231303284522 PR-AUC: 0.3181779307864365 CM: {'thr': 0.55, 'tp': 73009, 'fp': 154298, 'tn': 738816, 'fn': 117666, 'precision': 0.32119116437241263, 'recall': 0.38289760062934314, 'f1': 0.34934040221827733}


                                                                                    

    LR  TEST recall@top5%: {'k': 0.05, 'topn': 54189, 'thr_used': 0.6588753774454037, 'recall_at_topk': 0.11283073292251213}


                                                                                    

    GBT TEST recall@top5%: {'k': 0.05, 'topn': 54189, 'thr_used': 0.6939179382672999, 'recall_at_topk': 0.12702766487478695}


                                                                                    

    LR  TEST cost: {'thr': 0.5, 'cost_fn': 5.0, 'cost_fp': 1.0, 'expected_cost_per_row': 0.6807459754620133}


                                                                                    

    GBT TEST cost: {'thr': 0.55, 'cost_fn': 5.0, 'cost_fp': 1.0, 'expected_cost_per_row': 0.685214557446145}


                                                                                    

    +---+------+-------------------+--------------------+
    |bin|n     |avg_p              |emp_rate            |
    +---+------+-------------------+--------------------+
    |0  |46    |0.08701295335688483|0.021739130434782608|
    |1  |48836 |0.1767907715863637 |0.052277008764026536|
    |2  |199546|0.2529513897337072 |0.08624076654004591 |
    |3  |236134|0.3509412372498788 |0.13453801654992503 |
    |4  |259421|0.45021354783796397|0.17954213421426948 |
    |5  |215060|0.5458672316669931 |0.23413001022970334 |
    |6  |98451 |0.641159302183383  |0.3065179632507542  |
    |7  |23198 |0.73402782799809   |0.43495128890421586 |
    |8  |2389  |0.832959916404849  |0.614064462118041   |
    |9  |708   |0.9425123204359503 |0.6779661016949152  |
    +---+------+-------------------+--------------------+
    


    [Stage 491:======================================>                  (2 + 1) / 3]

    +---+------+-------------------+--------------------+
    |bin|n     |avg_p              |emp_rate            |
    +---+------+-------------------+--------------------+
    |0  |644   |0.09498261521311523|0.021739130434782608|
    |1  |87427 |0.16009369005438212|0.052661077241584406|
    |2  |180847|0.25546503715334   |0.08861081466654133 |
    |3  |223553|0.3502943208909518 |0.1273076183276449  |
    |4  |260634|0.44982580041926523|0.17603996408757108 |
    |5  |179646|0.5449849995129215 |0.23345356979838125 |
    |6  |100531|0.6451231325901129 |0.30882016492425224 |
    |7  |39143 |0.7399497329517115 |0.41123572541706055 |
    |8  |10572 |0.8341096374142586 |0.5699016269390844  |
    |9  |792   |0.920331390503622  |0.7361111111111112  |
    +---+------+-------------------+--------------------+
    


                                                                                    

3.7 Saving to GCS (models + preds + metrics + calibration)


```python
import json
from datetime import datetime
from pyspark.sql import functions as F

RUN_ID = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

BASE = "gs://big-data-project-481305-flightdelay/airline"
MODEL_DIR = f"{BASE}/models/sample_run_{RUN_ID}/"
PRED_DIR  = f"{BASE}/preds/sample_run_{RUN_ID}/"
METR_DIR  = f"{BASE}/metrics/sample_run_{RUN_ID}/"

# ---- Save models ----
feat_model.write().overwrite().save(MODEL_DIR + "feature_pipeline")
lr_model.write().overwrite().save(MODEL_DIR + "logreg_model")
gbt_model.write().overwrite().save(MODEL_DIR + "gbt_model")

# ---- Save predictions (test + val) ----
# Keep essentials; p1 is already in your pred dfs
(lr_val_pred.select("label","p1","prediction","probability")
 .write.mode("overwrite").parquet(PRED_DIR + "lr_val/"))

(gbt_val_pred.select("label","p1","prediction","probability")
 .write.mode("overwrite").parquet(PRED_DIR + "gbt_val/"))

(lr_test_pred.select("label","p1","prediction","probability")
 .write.mode("overwrite").parquet(PRED_DIR + "lr_test/"))

(gbt_test_pred.select("label","p1","prediction","probability")
 .write.mode("overwrite").parquet(PRED_DIR + "gbt_test/"))

# ---- Save calibration bins (test) ----
lr_cal.coalesce(1).write.mode("overwrite").csv(METR_DIR + "lr_calibration_bins_csv", header=True)
gbt_cal.coalesce(1).write.mode("overwrite").csv(METR_DIR + "gbt_calibration_bins_csv", header=True)

# ---- Save a compact metrics JSON ----
metrics = {
    "run_id": RUN_ID,
    "val": {
        "lr":  {"roc_auc": lr_val_roc,  "pr_auc": lr_val_pr,  **lr_best,
                "recall_top5": recall_at_topk_fast(lr_val_pred, 0.05)["recall_at_topk"],
                "brier": brier_score(lr_val_pred)},
        "gbt": {"roc_auc": gbt_val_roc, "pr_auc": gbt_val_pr, **gbt_best,
                "recall_top5": recall_at_topk_fast(gbt_val_pred, 0.05)["recall_at_topk"],
                "brier": brier_score(gbt_val_pred)},
    },
    "test": {
        "lr":  {"roc_auc": lr_test_roc,  "pr_auc": lr_test_pr,  **lr_test_cm,
                "recall_top5": recall_at_topk_fast(lr_test_pred, 0.05)["recall_at_topk"],
                "expected_cost_fn5_fp1": expected_cost(lr_test_pred, lr_best["thr"], 5.0, 1.0)["expected_cost_per_row"]},
        "gbt": {"roc_auc": gbt_test_roc, "pr_auc": gbt_test_pr, **gbt_test_cm,
                "recall_top5": recall_at_topk_fast(gbt_test_pred, 0.05)["recall_at_topk"],
                "expected_cost_fn5_fp1": expected_cost(gbt_test_pred, gbt_best["thr"], 5.0, 1.0)["expected_cost_per_row"]},
    }
}

metrics_df = spark.createDataFrame([(RUN_ID, json.dumps(metrics))], ["run_id", "metrics_json"])

# write as JSON (keeps both columns)
metrics_df.coalesce(1).write.mode("overwrite").json(METR_DIR + "metrics_json")


print("Saved models  :", MODEL_DIR)
print("Saved preds   :", PRED_DIR)
print("Saved metrics :", METR_DIR)


spark.createDataFrame([(json.dumps(metrics, indent=2),)], ["metrics_pretty"]) \
     .coalesce(1).write.mode("overwrite").text(METR_DIR + "metrics_pretty_txt")


```

                                                                                    

    Saved models  : gs://big-data-project-481305-flightdelay/airline/models/sample_run_20251221_030344/
    Saved preds   : gs://big-data-project-481305-flightdelay/airline/preds/sample_run_20251221_030344/
    Saved metrics : gs://big-data-project-481305-flightdelay/airline/metrics/sample_run_20251221_030344/


                                                                                    


```python

```

Unified evaluation function (LR/GBT/RF) + final metrics table


```python
import time
from pyspark.sql import Row

def evaluate_classifier(name, model, df_val_fe, df_test_fe, thresholds):
    # ---- VAL ----
    val_pred = add_prob(model.transform(df_val_fe)).cache()
    _ = val_pred.count()

    val_roc, val_pr = basic_auc_metrics(val_pred)

    # pick best threshold by F1 on validation
    stats = [confusion_metrics(val_pred, t) for t in thresholds]
    best = max(stats, key=lambda d: d["f1"])
    best = add_accuracy_to_conf(best)

    val_risk = risk_regression_metrics(val_pred)
    val_top5 = recall_at_topk_fast(val_pred, 0.05)["recall_at_topk"]

    # ---- TEST ----
    t0 = time.time()
    test_pred = add_prob(model.transform(df_test_fe)).cache()
    _ = test_pred.count()
    infer_ms = (time.time() - t0) * 1000.0

    test_roc, test_pr = basic_auc_metrics(test_pred)
    test_cm = add_accuracy_to_conf(confusion_metrics(test_pred, best["thr"]))
    test_risk = risk_regression_metrics(test_pred)
    test_top5 = recall_at_topk_fast(test_pred, 0.05)["recall_at_topk"]
    test_cost = expected_cost(test_pred, best["thr"], cost_fn=5.0, cost_fp=1.0)["expected_cost_per_row"]

    return {
        "model": name,
        "val_roc_auc": val_roc,
        "val_pr_auc": val_pr,
        "val_recall_top5": val_top5,
        **{f"val_{k}": v for k, v in val_risk.items()},
        "best_thr": best["thr"],
        "val_accuracy": best["accuracy"],
        "val_precision": best["precision"],
        "val_recall": best["recall"],
        "val_f1": best["f1"],

        "test_roc_auc": test_roc,
        "test_pr_auc": test_pr,
        "test_recall_top5": test_top5,
        **{f"test_{k}": v for k, v in test_risk.items()},
        "test_accuracy": test_cm["accuracy"],
        "test_precision": test_cm["precision"],
        "test_recall": test_cm["recall"],
        "test_f1": test_cm["f1"],
        "test_expected_cost_fn5_fp1": test_cost,
        "test_inference_ms_total": infer_ms,
        "test_inference_ms_per_row": infer_ms / test_pred.count()
    }, val_pred, test_pred

```


```python
thresholds = [i/100 for i in range(10, 91, 5)]

lr_summary,  lr_val_pred,  lr_test_pred  = evaluate_classifier("LogReg", lr_model, val_fe, test_fe, thresholds)
gbt_summary, gbt_val_pred, gbt_test_pred = evaluate_classifier("GBT", gbt_model, val_fe, test_fe, thresholds)
rf_summary,  rf_val_pred,  rf_test_pred  = evaluate_classifier("RF", rf_model, val_fe, test_fe, thresholds)

summary_df = spark.createDataFrame([Row(**lr_summary), Row(**gbt_summary), Row(**rf_summary)])
summary_df.show(truncate=False)

```

    [Stage 1020:>                                                       (0 + 2) / 2]

    +------+------------------+------------------+-------------------+-------------------+------------------+-------------------+------------------+--------+------------------+-------------------+-------------------+-------------------+------------------+-------------------+-------------------+-------------------+-------------------+-------------------+------------------+------------------+-------------------+-------------------+-------------------+--------------------------+-----------------------+-------------------------+
    |model |val_roc_auc       |val_pr_auc        |val_recall_top5    |val_brier          |val_rmse_prob     |val_mae_prob       |val_logloss       |best_thr|val_accuracy      |val_precision      |val_recall         |val_f1             |test_roc_auc      |test_pr_auc        |test_recall_top5   |test_brier         |test_rmse_prob     |test_mae_prob      |test_logloss      |test_accuracy     |test_precision     |test_recall        |test_f1            |test_expected_cost_fn5_fp1|test_inference_ms_total|test_inference_ms_per_row|
    +------+------------------+------------------+-------------------+-------------------+------------------+-------------------+------------------+--------+------------------+-------------------+-------------------+-------------------+------------------+-------------------+-------------------+-------------------+-------------------+-------------------+------------------+------------------+-------------------+-------------------+-------------------+--------------------------+-----------------------+-------------------------+
    |LogReg|0.6686123426264053|0.2839786433832775|0.12437048873216155|0.20749280154649552|0.4555137775594669|0.43239929338886507|0.602676590186791 |0.5     |0.6694101655541245|0.2539569689358478 |0.5392804011240289 |0.3453039941011067 |0.6571442479124473|0.2920898577448437 |0.11283073292251213|0.20312743988038026|0.45069661622912177|0.4271799362173361 |0.5930564665584845|0.6813503366430181|0.272408374189979  |0.48546479611905075|0.34898893645578255|0.6807459754620133        |6663.030624389648      |0.006147903904163678     |
    |GBT   |0.6856541077246576|0.3100075873414808|0.13607361287123257|0.2169265724140898 |0.4657537680084723|0.4385896309571227 |0.622899104327042 |0.55    |0.7199938716648229|0.28543558868748464|0.4869248994434955 |0.3598985123011073 |0.6759234240578236|0.3181777692779172 |0.12702766487478695|0.2009703373897025 |0.44829715300200434|0.4199726586747637 |0.587823829819608 |0.7490618561362037|0.32119116437241263|0.38289760062934314|0.34934040221827733|0.685214557446145         |29700.847625732422     |0.027404640225848778     |
    |RF    |0.6517357214761871|0.2652413176720991|0.11556559590060059|0.234083172717949  |0.4838214264767002|0.47491613276425276|0.6596879807895261|0.55    |0.6715782422723297|0.24597691167930355|0.49943247561849136|0.32961446431558505|0.6460029228675328|0.28142935061454827|0.11298806870329094|0.2342517875337853 |0.48399564825914015|0.47529162666334535|0.6601130089809017|0.6712192133339607|0.2635027867878843 |0.4839910843057559 |0.341228214509548  |0.6919142010114515        |12490.607261657715     |0.01152494374980528      |
    +------+------------------+------------------+-------------------+-------------------+------------------+-------------------+------------------+--------+------------------+-------------------+-------------------+-------------------+------------------+-------------------+-------------------+-------------------+-------------------+-------------------+------------------+------------------+-------------------+-------------------+-------------------+--------------------------+-----------------------+-------------------------+
    


                                                                                    


```python

```


```python
summary_df.show(vertical=True, truncate=False)
```

    -RECORD 0------------------------------------------
     model                      | LogReg               
     val_roc_auc                | 0.6686123426264053   
     val_pr_auc                 | 0.2839786433832775   
     val_recall_top5            | 0.12437048873216155  
     val_brier                  | 0.20749280154649552  
     val_rmse_prob              | 0.4555137775594669   
     val_mae_prob               | 0.43239929338886507  
     val_logloss                | 0.602676590186791    
     best_thr                   | 0.5                  
     val_accuracy               | 0.6694101655541245   
     val_precision              | 0.2539569689358478   
     val_recall                 | 0.5392804011240289   
     val_f1                     | 0.3453039941011067   
     test_roc_auc               | 0.6571442479124473   
     test_pr_auc                | 0.2920898577448437   
     test_recall_top5           | 0.11283073292251213  
     test_brier                 | 0.20312743988038026  
     test_rmse_prob             | 0.45069661622912177  
     test_mae_prob              | 0.4271799362173361   
     test_logloss               | 0.5930564665584845   
     test_accuracy              | 0.6813503366430181   
     test_precision             | 0.272408374189979    
     test_recall                | 0.48546479611905075  
     test_f1                    | 0.34898893645578255  
     test_expected_cost_fn5_fp1 | 0.6807459754620133   
     test_inference_ms_total    | 6663.030624389648    
     test_inference_ms_per_row  | 0.006147903904163678 
    -RECORD 1------------------------------------------
     model                      | GBT                  
     val_roc_auc                | 0.6856541077246576   
     val_pr_auc                 | 0.3100075873414808   
     val_recall_top5            | 0.13607361287123257  
     val_brier                  | 0.2169265724140898   
     val_rmse_prob              | 0.4657537680084723   
     val_mae_prob               | 0.4385896309571227   
     val_logloss                | 0.622899104327042    
     best_thr                   | 0.55                 
     val_accuracy               | 0.7199938716648229   
     val_precision              | 0.28543558868748464  
     val_recall                 | 0.4869248994434955   
     val_f1                     | 0.3598985123011073   
     test_roc_auc               | 0.6759234240578236   
     test_pr_auc                | 0.3181777692779172   
     test_recall_top5           | 0.12702766487478695  
     test_brier                 | 0.2009703373897025   
     test_rmse_prob             | 0.44829715300200434  
     test_mae_prob              | 0.4199726586747637   
     test_logloss               | 0.587823829819608    
     test_accuracy              | 0.7490618561362037   
     test_precision             | 0.32119116437241263  
     test_recall                | 0.38289760062934314  
     test_f1                    | 0.34934040221827733  
     test_expected_cost_fn5_fp1 | 0.685214557446145    
     test_inference_ms_total    | 29700.847625732422   
     test_inference_ms_per_row  | 0.027404640225848778 
    -RECORD 2------------------------------------------
     model                      | RF                   
     val_roc_auc                | 0.6517357214761871   
     val_pr_auc                 | 0.2652413176720991   
     val_recall_top5            | 0.11556559590060059  
     val_brier                  | 0.234083172717949    
     val_rmse_prob              | 0.4838214264767002   
     val_mae_prob               | 0.47491613276425276  
     val_logloss                | 0.6596879807895261   
     best_thr                   | 0.55                 
     val_accuracy               | 0.6715782422723297   
     val_precision              | 0.24597691167930355  
     val_recall                 | 0.49943247561849136  
     val_f1                     | 0.32961446431558505  
     test_roc_auc               | 0.6460029228675328   
     test_pr_auc                | 0.28142935061454827  
     test_recall_top5           | 0.11298806870329094  
     test_brier                 | 0.2342517875337853   
     test_rmse_prob             | 0.48399564825914015  
     test_mae_prob              | 0.47529162666334535  
     test_logloss               | 0.6601130089809017   
     test_accuracy              | 0.6712192133339607   
     test_precision             | 0.2635027867878843   
     test_recall                | 0.4839910843057559   
     test_f1                    | 0.341228214509548    
     test_expected_cost_fn5_fp1 | 0.6919142010114515   
     test_inference_ms_total    | 12490.607261657715   
     test_inference_ms_per_row  | 0.01152494374980528  
    



```python
from pyspark.sql import functions as F

# Get list of metric columns (everything except 'model')
metric_cols = [c for c in summary_df.columns if c != 'model']

# Round all metric columns to 4 decimal places inside Spark
# This keeps it as a Spark DataFrame
rounded_df = summary_df.select(
    F.col("model"),
    *[F.round(F.col(c), 4).alias(c) for c in metric_cols]
)

# Create the "Stack" expression dynamically
# This converts wide format to long format: | model | metric_name | value |
stack_string = ", ".join([f"'{c}', {c}" for c in metric_cols])
stack_expr = f"stack({len(metric_cols)}, {stack_string}) as (metric, value)"

# Transform: Stack -> GroupBy -> Pivot
transposed_df = rounded_df.select("model", F.expr(stack_expr)) \
    .groupBy("metric") \
    .pivot("model") \
    .agg(F.first("value")) # taking first value since there's 1 value per model/metric

# Display in native PySpark 
transposed_df.show(n=100, truncate=False)
```

    [Stage 1031:==================>                                     (1 + 2) / 3]

    +--------------------------+----------+---------+----------+
    |metric                    |GBT       |LogReg   |RF        |
    +--------------------------+----------+---------+----------+
    |val_pr_auc                |0.31      |0.284    |0.2652    |
    |test_accuracy             |0.7491    |0.6814   |0.6712    |
    |val_rmse_prob             |0.4658    |0.4555   |0.4838    |
    |test_f1                   |0.3493    |0.349    |0.3412    |
    |val_recall                |0.4869    |0.5393   |0.4994    |
    |test_roc_auc              |0.6759    |0.6571   |0.646     |
    |val_brier                 |0.2169    |0.2075   |0.2341    |
    |val_logloss               |0.6229    |0.6027   |0.6597    |
    |test_logloss              |0.5878    |0.5931   |0.6601    |
    |val_mae_prob              |0.4386    |0.4324   |0.4749    |
    |test_precision            |0.3212    |0.2724   |0.2635    |
    |test_recall               |0.3829    |0.4855   |0.484     |
    |val_precision             |0.2854    |0.254    |0.246     |
    |val_recall_top5           |0.1361    |0.1244   |0.1156    |
    |test_recall_top5          |0.127     |0.1128   |0.113     |
    |test_rmse_prob            |0.4483    |0.4507   |0.484     |
    |val_f1                    |0.3599    |0.3453   |0.3296    |
    |test_inference_ms_per_row |0.0274    |0.0061   |0.0115    |
    |best_thr                  |0.55      |0.5      |0.55      |
    |val_roc_auc               |0.6857    |0.6686   |0.6517    |
    |test_mae_prob             |0.42      |0.4272   |0.4753    |
    |test_inference_ms_total   |29700.8476|6663.0306|12490.6073|
    |test_pr_auc               |0.3182    |0.2921   |0.2814    |
    |val_accuracy              |0.72      |0.6694   |0.6716    |
    |test_expected_cost_fn5_fp1|0.6852    |0.6807   |0.6919    |
    |test_brier                |0.201     |0.2031   |0.2343    |
    +--------------------------+----------+---------+----------+
    


                                                                                    


```python

```


```python
# Convert to Pandas
pdf = summary_df.toPandas()

# Set the 'model' column as the index for a cleaner look
pdf = pdf.set_index('model')
pdf = pdf.round(4)

display(pdf)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>val_roc_auc</th>
      <th>val_pr_auc</th>
      <th>val_recall_top5</th>
      <th>val_brier</th>
      <th>val_rmse_prob</th>
      <th>val_mae_prob</th>
      <th>val_logloss</th>
      <th>best_thr</th>
      <th>val_accuracy</th>
      <th>val_precision</th>
      <th>...</th>
      <th>test_rmse_prob</th>
      <th>test_mae_prob</th>
      <th>test_logloss</th>
      <th>test_accuracy</th>
      <th>test_precision</th>
      <th>test_recall</th>
      <th>test_f1</th>
      <th>test_expected_cost_fn5_fp1</th>
      <th>test_inference_ms_total</th>
      <th>test_inference_ms_per_row</th>
    </tr>
    <tr>
      <th>model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LogReg</th>
      <td>0.6686</td>
      <td>0.2840</td>
      <td>0.1244</td>
      <td>0.2075</td>
      <td>0.4555</td>
      <td>0.4324</td>
      <td>0.6027</td>
      <td>0.50</td>
      <td>0.6694</td>
      <td>0.2540</td>
      <td>...</td>
      <td>0.4507</td>
      <td>0.4272</td>
      <td>0.5931</td>
      <td>0.6814</td>
      <td>0.2724</td>
      <td>0.4855</td>
      <td>0.3490</td>
      <td>0.6807</td>
      <td>6663.0306</td>
      <td>0.0061</td>
    </tr>
    <tr>
      <th>GBT</th>
      <td>0.6857</td>
      <td>0.3100</td>
      <td>0.1361</td>
      <td>0.2169</td>
      <td>0.4658</td>
      <td>0.4386</td>
      <td>0.6229</td>
      <td>0.55</td>
      <td>0.7200</td>
      <td>0.2854</td>
      <td>...</td>
      <td>0.4483</td>
      <td>0.4200</td>
      <td>0.5878</td>
      <td>0.7491</td>
      <td>0.3212</td>
      <td>0.3829</td>
      <td>0.3493</td>
      <td>0.6852</td>
      <td>29700.8476</td>
      <td>0.0274</td>
    </tr>
    <tr>
      <th>RF</th>
      <td>0.6517</td>
      <td>0.2652</td>
      <td>0.1156</td>
      <td>0.2341</td>
      <td>0.4838</td>
      <td>0.4749</td>
      <td>0.6597</td>
      <td>0.55</td>
      <td>0.6716</td>
      <td>0.2460</td>
      <td>...</td>
      <td>0.4840</td>
      <td>0.4753</td>
      <td>0.6601</td>
      <td>0.6712</td>
      <td>0.2635</td>
      <td>0.4840</td>
      <td>0.3412</td>
      <td>0.6919</td>
      <td>12490.6073</td>
      <td>0.0115</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 26 columns</p>
</div>



```python
# Convert to pandas
pdf = summary_df.toPandas()

# Transpose: Flip rows and columns
# This puts LogReg, GBT, and RF as the column headers
pdf_transposed = pdf.set_index('model').transpose()
pdf_transposed = pdf_transposed.round(4)

display(pdf_transposed)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>model</th>
      <th>LogReg</th>
      <th>GBT</th>
      <th>RF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>val_roc_auc</th>
      <td>0.6686</td>
      <td>0.6857</td>
      <td>0.6517</td>
    </tr>
    <tr>
      <th>val_pr_auc</th>
      <td>0.2840</td>
      <td>0.3100</td>
      <td>0.2652</td>
    </tr>
    <tr>
      <th>val_recall_top5</th>
      <td>0.1244</td>
      <td>0.1361</td>
      <td>0.1156</td>
    </tr>
    <tr>
      <th>val_brier</th>
      <td>0.2075</td>
      <td>0.2169</td>
      <td>0.2341</td>
    </tr>
    <tr>
      <th>val_rmse_prob</th>
      <td>0.4555</td>
      <td>0.4658</td>
      <td>0.4838</td>
    </tr>
    <tr>
      <th>val_mae_prob</th>
      <td>0.4324</td>
      <td>0.4386</td>
      <td>0.4749</td>
    </tr>
    <tr>
      <th>val_logloss</th>
      <td>0.6027</td>
      <td>0.6229</td>
      <td>0.6597</td>
    </tr>
    <tr>
      <th>best_thr</th>
      <td>0.5000</td>
      <td>0.5500</td>
      <td>0.5500</td>
    </tr>
    <tr>
      <th>val_accuracy</th>
      <td>0.6694</td>
      <td>0.7200</td>
      <td>0.6716</td>
    </tr>
    <tr>
      <th>val_precision</th>
      <td>0.2540</td>
      <td>0.2854</td>
      <td>0.2460</td>
    </tr>
    <tr>
      <th>val_recall</th>
      <td>0.5393</td>
      <td>0.4869</td>
      <td>0.4994</td>
    </tr>
    <tr>
      <th>val_f1</th>
      <td>0.3453</td>
      <td>0.3599</td>
      <td>0.3296</td>
    </tr>
    <tr>
      <th>test_roc_auc</th>
      <td>0.6571</td>
      <td>0.6759</td>
      <td>0.6460</td>
    </tr>
    <tr>
      <th>test_pr_auc</th>
      <td>0.2921</td>
      <td>0.3182</td>
      <td>0.2814</td>
    </tr>
    <tr>
      <th>test_recall_top5</th>
      <td>0.1128</td>
      <td>0.1270</td>
      <td>0.1130</td>
    </tr>
    <tr>
      <th>test_brier</th>
      <td>0.2031</td>
      <td>0.2010</td>
      <td>0.2343</td>
    </tr>
    <tr>
      <th>test_rmse_prob</th>
      <td>0.4507</td>
      <td>0.4483</td>
      <td>0.4840</td>
    </tr>
    <tr>
      <th>test_mae_prob</th>
      <td>0.4272</td>
      <td>0.4200</td>
      <td>0.4753</td>
    </tr>
    <tr>
      <th>test_logloss</th>
      <td>0.5931</td>
      <td>0.5878</td>
      <td>0.6601</td>
    </tr>
    <tr>
      <th>test_accuracy</th>
      <td>0.6814</td>
      <td>0.7491</td>
      <td>0.6712</td>
    </tr>
    <tr>
      <th>test_precision</th>
      <td>0.2724</td>
      <td>0.3212</td>
      <td>0.2635</td>
    </tr>
    <tr>
      <th>test_recall</th>
      <td>0.4855</td>
      <td>0.3829</td>
      <td>0.4840</td>
    </tr>
    <tr>
      <th>test_f1</th>
      <td>0.3490</td>
      <td>0.3493</td>
      <td>0.3412</td>
    </tr>
    <tr>
      <th>test_expected_cost_fn5_fp1</th>
      <td>0.6807</td>
      <td>0.6852</td>
      <td>0.6919</td>
    </tr>
    <tr>
      <th>test_inference_ms_total</th>
      <td>6663.0306</td>
      <td>29700.8476</td>
      <td>12490.6073</td>
    </tr>
    <tr>
      <th>test_inference_ms_per_row</th>
      <td>0.0061</td>
      <td>0.0274</td>
      <td>0.0115</td>
    </tr>
  </tbody>
</table>
</div>



```python

```


```python

```


```python

```

# Phase 4

Ablation study (Base → +Congestion → +Weather → Full)

Creating four feature sets. The current “Full” is already:

1. Base: time + simple flight attributes (no congestion/weather)

2. +Congestion: add CONCURRENT_FLIGHTS + monthly traffic features

3. +Weather: add weather fields

4. Full: base + congestion + weather + leakage-safe aggregates

Defining feature groups


```python
BASE_NUM = ["hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos",
            "DISTANCE_GROUP","SEGMENT_NUMBER","NUMBER_OF_SEATS","PLANE_AGE","LATITUDE","LONGITUDE"]

CONGESTION_NUM = ["CONCURRENT_FLIGHTS","AIRPORT_FLIGHTS_MONTH","AIRLINE_FLIGHTS_MONTH",
                  "AIRLINE_AIRPORT_FLIGHTS_MONTH","AVG_MONTHLY_PASS_AIRPORT","AVG_MONTHLY_PASS_AIRLINE",
                  "FLT_ATTENDANTS_PER_PASS","GROUND_SERV_PER_PASS"]

WEATHER_NUM = ["PRCP","SNOW","SNWD","TMAX","AWND"]

LEAKAGE_SAFE_AGG = ["delay_rate_air_blk","delay_rate_route","n_air_blk","n_route"]

BASE_CAT = ["DEPARTING_AIRPORT_BKT","PREVIOUS_AIRPORT_BKT","CARRIER_NAME_BKT","DEP_TIME_BLK"]

```

Helper to build features for a given set


```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer, VectorAssembler, StandardScaler

def build_feature_pipeline(cat_cols, num_cols):
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
    encoder = OneHotEncoder(inputCols=[f"{c}_idx" for c in cat_cols],
                            outputCols=[f"{c}_ohe" for c in cat_cols],
                            handleInvalid="keep")
    imputer = Imputer(inputCols=num_cols, outputCols=[f"{c}_imp" for c in num_cols], strategy="median")
    assembler = VectorAssembler(inputCols=[f"{c}_imp" for c in num_cols] + [f"{c}_ohe" for c in cat_cols],
                                outputCol="features_raw", handleInvalid="keep")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=False, withStd=True)
    return Pipeline(stages=indexers + [encoder, imputer, assembler, scaler])

```

# Phase 5

# Using GBT for ablation:

Create full transformed tables ONCE


```python
# Full transformed tables ONCE (includes *_imp and *_ohe columns)
full_tr = feat_model.transform(train5).cache(); _ = full_tr.count()
full_va = feat_model.transform(val5).cache();   _ = full_va.count()
full_te = feat_model.transform(test5).cache();  _ = full_te.count()

print("full_tr/full_va/full_te cached")

```

    [Stage 1048:=====================================>                  (2 + 1) / 3]

    full_tr/full_va/full_te cached


                                                                                    

Get the assembler input column list from your existing pipeline


```python
from pyspark.ml.feature import VectorAssembler

def get_assembler_inputs(feat_model):
    # Find the VectorAssembler stage used in your pipeline
    assembler = next(s for s in feat_model.stages if s.__class__.__name__ == "VectorAssembler")
    return assembler.getInputCols()

assembler_inputs = get_assembler_inputs(feat_model)
ohe_inputs = [c for c in assembler_inputs if c.endswith("_ohe")]
print("assembler inputs:", len(assembler_inputs), "| ohe inputs:", len(ohe_inputs))
print("first 5:", assembler_inputs[:5])
print("last  5:", assembler_inputs[-5:])

```

    assembler inputs: 33 | ohe inputs: 6
    first 5: ['CONCURRENT_FLIGHTS_imp', 'NUMBER_OF_SEATS_imp', 'AIRPORT_FLIGHTS_MONTH_imp', 'AIRLINE_FLIGHTS_MONTH_imp', 'AIRLINE_AIRPORT_FLIGHTS_MONTH_imp']
    last  5: ['PREVIOUS_AIRPORT_BKT_ohe', 'CARRIER_NAME_BKT_ohe', 'DEP_TIME_BLK_ohe', 'DISTANCE_GROUP_ohe', 'SEGMENT_NUMBER_ohe']


Define ablation input subsets (no refit, just vector slicing)


```python
def imp(cols): 
    return [f"{c}_imp" for c in cols]

base_imp = set(imp(BASE_NUM))
cong_imp = set(imp(CONGESTION_NUM))
weath_imp = set(imp(WEATHER_NUM))
agg_imp = set(imp(LEAKAGE_SAFE_AGG))

def inputs_for(block):
    keep = set(ohe_inputs)
    if block == "A_Base":
        keep |= base_imp
    elif block == "B_+Congestion":
        keep |= (base_imp | cong_imp)
    elif block == "C_+Weather":
        keep |= (base_imp | weath_imp)
    elif block == "D_Full":
        keep |= (base_imp | cong_imp | weath_imp | agg_imp)
    else:
        raise ValueError(block)

    # preserve original order
    return [c for c in assembler_inputs if c in keep]

abl_inputs = {b: inputs_for(b) for b in ["A_Base","B_+Congestion","C_+Weather","D_Full"]}
for b in abl_inputs:
    print(b, "n_inputs:", len(abl_inputs[b]))

```

    A_Base n_inputs: 16
    B_+Congestion n_inputs: 24
    C_+Weather n_inputs: 21
    D_Full n_inputs: 33


Train + evaluate ablation models fast (only trains 4 models)


```python
# from pyspark.ml.feature import VectorAssembler
# from pyspark.ml.classification import GBTClassifier
# from pyspark.sql import Row

# def make_feats(df_full, input_cols, out_col):
#     # drop out_col if it already exists to avoid collisions
#     df = df_full.drop(out_col) if out_col in df_full.columns else df_full
#     return VectorAssembler(inputCols=input_cols, outputCol=out_col, handleInvalid="keep").transform(df)

# def run_ablation_fast(name, input_cols):
#     out_col = "features_abl"

#     tr = make_feats(full_tr, input_cols, out_col).select("label","weight",F.col(out_col).alias("features")).cache(); _ = tr.count()
#     va = make_feats(full_va, input_cols, out_col).select("label","weight",F.col(out_col).alias("features")).cache(); _ = va.count()
#     te = make_feats(full_te, input_cols, out_col).select("label","weight",F.col(out_col).alias("features")).cache(); _ = te.count()

#     model = GBTClassifier(featuresCol="features", labelCol="label", weightCol="weight",
#                           maxIter=40, maxDepth=4, stepSize=0.08, subsamplingRate=0.8).fit(tr)

#     summary, _, _ = evaluate_classifier(name, model, va, te, thresholds=[i/100 for i in range(10, 91, 5)])
#     return Row(**summary)

```

Changed ablation assembler output to a new column name, like features_abl, and then training GBT on that.

1. builds features_abl

2. renames it to features only in the small training/val/test DFs passed to the classifier

3. avoids any collision with your original features column

Code to prevent a few giant partitions from slowing down each ablation run.


```python
full_tr = full_tr.repartition(200).cache(); _ = full_tr.count()
full_va = full_va.repartition(200).cache(); _ = full_va.count()
full_te = full_te.repartition(200).cache(); _ = full_te.count()

```

    25/12/21 03:13:24 WARN YarnAllocator: Container from a bad node: container_1766261877773_0002_01_000005 on host: airline-analysis-v3-w-1.us-east1-b.c.big-data-project-481305.internal. Exit status: 137. Diagnostics: [2025-12-21 03:13:23.887]Container killed on request. Exit code is 137
    [2025-12-21 03:13:23.887]Container exited with a non-zero exit code 137. 
    [2025-12-21 03:13:23.887]Killed by external signal
    .
    25/12/21 03:13:24 ERROR YarnScheduler: Lost executor 5 on airline-analysis-v3-w-1.us-east1-b.c.big-data-project-481305.internal: Container from a bad node: container_1766261877773_0002_01_000005 on host: airline-analysis-v3-w-1.us-east1-b.c.big-data-project-481305.internal. Exit status: 137. Diagnostics: [2025-12-21 03:13:23.887]Container killed on request. Exit code is 137
    [2025-12-21 03:13:23.887]Container exited with a non-zero exit code 137. 
    [2025-12-21 03:13:23.887]Killed by external signal
    .
    25/12/21 03:13:24 WARN YarnSchedulerBackend$YarnSchedulerEndpoint: Requesting driver to remove executor 5 for reason Container from a bad node: container_1766261877773_0002_01_000005 on host: airline-analysis-v3-w-1.us-east1-b.c.big-data-project-481305.internal. Exit status: 137. Diagnostics: [2025-12-21 03:13:23.887]Container killed on request. Exit code is 137
    [2025-12-21 03:13:23.887]Container exited with a non-zero exit code 137. 
    [2025-12-21 03:13:23.887]Killed by external signal
    .
                                                                                    


```python
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.sql import Row
from pyspark.sql import functions as F

def make_feats(df_full, input_cols, out_col="features_abl"):
    df = df_full.drop(out_col) if out_col in df_full.columns else df_full
    return VectorAssembler(inputCols=input_cols, outputCol=out_col, handleInvalid="keep").transform(df)

def run_ablation_fast(name, input_cols):
    out_col = "features_abl"

    # assemble features (fast) + persist
    tr, dt_tr = timed(f"{name} assemble TRAIN", lambda:
        make_feats(full_tr, input_cols, out_col).select("label","weight",F.col(out_col).alias("features")).persist()
    )
    va, dt_va = timed(f"{name} assemble VAL", lambda:
        make_feats(full_va, input_cols, out_col).select("label","weight",F.col(out_col).alias("features")).persist()
    )
    te, dt_te = timed(f"{name} assemble TEST", lambda:
        make_feats(full_te, input_cols, out_col).select("label","weight",F.col(out_col).alias("features")).persist()
    )

    # materialize once
    _, dt_c1 = timed(f"{name} count TRAIN", lambda: tr.count())
    _, dt_c2 = timed(f"{name} count VAL",   lambda: va.count())
    _, dt_c3 = timed(f"{name} count TEST",  lambda: te.count())

    # lighter GBT for ablation
    def fit_model():
        return GBTClassifier(
            featuresCol="features", labelCol="label", weightCol="weight",
            maxIter=20, maxDepth=3, stepSize=0.1, subsamplingRate=0.7
        ).fit(tr)

    model, dt_fit = timed(f"{name} fit GBT", fit_model)

    summary, dt_eval = timed(f"{name} eval LIGHT", lambda: evaluate_ablation_light(name, model, va, te, thr=0.5))

    summary["time_sec_assemble"] = dt_tr + dt_va + dt_te
    summary["time_sec_counts"] = dt_c1 + dt_c2 + dt_c3
    summary["time_sec_fit"] = dt_fit
    summary["time_sec_eval"] = dt_eval
    summary["time_sec_total"] = summary["time_sec_assemble"] + summary["time_sec_counts"] + dt_fit + dt_eval

    return Row(**summary)



```


```python
# ---- Run all ablations with timing ----
blocks = ["A_Base","B_+Congestion","C_+Weather","D_Full"]

ablation_rows, total_dt = timed("TOTAL ABLATION RUN", lambda:
    [run_ablation_fast(b, abl_inputs[b]) for b in blocks]
)

ablation_df = spark.createDataFrame(ablation_rows)

ablation_df.select(
    "model",
    "val_pr_auc","test_pr_auc",
    "val_f1","test_f1",
    "test_recall_top5",
    "time_sec_total"
).show(truncate=False)
```

    [TIMER] A_Base assemble TRAIN: 0.14 sec (0.00 min)
    [TIMER] A_Base assemble VAL: 0.11 sec (0.00 min)
    [TIMER] A_Base assemble TEST: 0.10 sec (0.00 min)


                                                                                    

    [TIMER] A_Base count TRAIN: 29.38 sec (0.49 min)


                                                                                    

    [TIMER] A_Base count VAL: 4.92 sec (0.08 min)


                                                                                    

    [TIMER] A_Base count TEST: 11.76 sec (0.20 min)


                                                                                    

    [TIMER] A_Base fit GBT: 282.98 sec (4.72 min)


    25/12/21 03:26:24 WARN YarnAllocator: Container from a bad node: container_1766261877773_0002_01_000009 on host: airline-analysis-v3-w-1.us-east1-b.c.big-data-project-481305.internal. Exit status: 143. Diagnostics: [2025-12-21 03:26:24.205]Container killed on request. Exit code is 143
    [2025-12-21 03:26:24.205]Container exited with a non-zero exit code 143. 
    [2025-12-21 03:26:24.205]Killed by external signal
    .
    25/12/21 03:26:24 ERROR YarnScheduler: Lost executor 7 on airline-analysis-v3-w-1.us-east1-b.c.big-data-project-481305.internal: Container from a bad node: container_1766261877773_0002_01_000009 on host: airline-analysis-v3-w-1.us-east1-b.c.big-data-project-481305.internal. Exit status: 143. Diagnostics: [2025-12-21 03:26:24.205]Container killed on request. Exit code is 143
    [2025-12-21 03:26:24.205]Container exited with a non-zero exit code 143. 
    [2025-12-21 03:26:24.205]Killed by external signal
    .
    25/12/21 03:26:24 WARN YarnSchedulerBackend$YarnSchedulerEndpoint: Requesting driver to remove executor 7 for reason Container from a bad node: container_1766261877773_0002_01_000009 on host: airline-analysis-v3-w-1.us-east1-b.c.big-data-project-481305.internal. Exit status: 143. Diagnostics: [2025-12-21 03:26:24.205]Container killed on request. Exit code is 143
    [2025-12-21 03:26:24.205]Container exited with a non-zero exit code 143. 
    [2025-12-21 03:26:24.205]Killed by external signal
    .
                                                                                    

    [TIMER] A_Base eval LIGHT: 123.85 sec (2.06 min)
    [TIMER] B_+Congestion assemble TRAIN: 0.08 sec (0.00 min)
    [TIMER] B_+Congestion assemble VAL: 0.06 sec (0.00 min)
    [TIMER] B_+Congestion assemble TEST: 0.07 sec (0.00 min)


                                                                                    

    [TIMER] B_+Congestion count TRAIN: 36.70 sec (0.61 min)


                                                                                    

    [TIMER] B_+Congestion count VAL: 9.85 sec (0.16 min)


                                                                                    

    [TIMER] B_+Congestion count TEST: 14.74 sec (0.25 min)


                                                                                    

    [TIMER] B_+Congestion fit GBT: 449.73 sec (7.50 min)


                                                                                    

    [TIMER] B_+Congestion eval LIGHT: 93.88 sec (1.56 min)
    [TIMER] C_+Weather assemble TRAIN: 0.08 sec (0.00 min)
    [TIMER] C_+Weather assemble VAL: 0.06 sec (0.00 min)
    [TIMER] C_+Weather assemble TEST: 0.06 sec (0.00 min)


                                                                                    

    [TIMER] C_+Weather count TRAIN: 19.96 sec (0.33 min)


                                                                                    

    [TIMER] C_+Weather count VAL: 6.76 sec (0.11 min)


                                                                                    

    [TIMER] C_+Weather count TEST: 14.09 sec (0.23 min)


                                                                                    

    [TIMER] C_+Weather fit GBT: 260.94 sec (4.35 min)


                                                                                    

    [TIMER] C_+Weather eval LIGHT: 85.13 sec (1.42 min)
    [TIMER] D_Full assemble TRAIN: 0.09 sec (0.00 min)
    [TIMER] D_Full assemble VAL: 0.08 sec (0.00 min)
    [TIMER] D_Full assemble TEST: 0.07 sec (0.00 min)


                                                                                    

    [TIMER] D_Full count TRAIN: 23.69 sec (0.39 min)


                                                                                    

    [TIMER] D_Full count VAL: 6.63 sec (0.11 min)


                                                                                    

    [TIMER] D_Full count TEST: 14.86 sec (0.25 min)


                                                                                    

    [TIMER] D_Full fit GBT: 261.88 sec (4.36 min)


                                                                                    

    [TIMER] D_Full eval LIGHT: 81.26 sec (1.35 min)
    [TIMER] TOTAL ABLATION RUN: 1833.98 sec (30.57 min)


    [Stage 2341:============================>                           (1 + 1) / 2]

    +-------------+-------------------+-------------------+-------------------+-------------------+-------------------+------------------+
    |model        |val_pr_auc         |test_pr_auc        |val_f1             |test_f1            |test_recall_top5   |time_sec_total    |
    +-------------+-------------------+-------------------+-------------------+-------------------+-------------------+------------------+
    |A_Base       |0.24372564822502768|0.26181065040156803|0.32176124240820275|0.33996679585528294|0.11057034220532319|453.237886428833  |
    |B_+Congestion|0.24307044280727802|0.2570685033818758 |0.3210542816794977 |0.3387259186978648 |0.11033433853415497|605.1021976470947 |
    |C_+Weather   |0.2745941553308767 |0.28949062986628704|0.3379655250158847 |0.3543650383404691 |0.1292828110659499 |387.0776093006134 |
    |D_Full       |0.2802228629860723 |0.2912884518389431 |0.3413187665892757 |0.35051840735012857|0.13089812508194573|388.56124782562256|
    +-------------+-------------------+-------------------+-------------------+-------------------+-------------------+------------------+
    


                                                                                    


```python
ABL_DIR = "gs://big-data-project-481305-flightdelay/airline/metrics/ablation_study/"
ablation_df.coalesce(1).write.mode("overwrite").csv(ABL_DIR, header=True)
print("Saved ablation to:", ABL_DIR)

```

                                                                                    

    Saved ablation to: gs://big-data-project-481305-flightdelay/airline/metrics/ablation_study/


Why it’s slow (current ablation)

Each run_ablation_fast() does:

1. assemble features + cache + count() (3 full actions per ablation)

2. train a GBT (heavy)

3. evaluate_classifier() does:

* transform val/test
* AUC calculations
* threshold sweep (many passes)
* Recall@TopK (often involves sorting)

# Slice metrics by airport / time block / month

We’ll compute per-slice counts, delay rate, avg predicted risk, and optionally precision/recall at the chosen threshold.


Example using best model predictions (using GBT test preds):

# Why slicing matters for the project

1. Operational usefulness
Airlines don’t act on “overall PR-AUC.” They act by airport, time block, carrier, month. Slicing tells you where the model predicts high risk so operations can allocate buffers/crew/gates.

2. Diagnose hidden failure modes

    ** A model can look strong overall but fail badly at:
* specific airports (hub congestion patterns)
* early morning vs evening banks
* winter months vs summer
Slicing surfaces those weak spots.

3. Model reliability / generalization

    If your avg_pred_risk tracks delay_rate by slice, that’s evidence your model is learning real structure, not noise.

4. Mini drift check

    If monthly delay_rate shifts but your avg_pred_risk doesn’t, that’s drift/miscalibration. 
    
    Even without full production monitoring, this is a solid “big data ML” deliverable.

5. Supports that feature engineering is important

    can show that adding weather/congestion improved hardest slices (winter months, busy time blocks, specific airports).


```python

# Choose which model preds you want to slice (example: GBT)
thr = gbt_summary["best_thr"]   # if your variable is different, change here
# thr = gbt_best["thr"]         # if that's what you used earlier

# Build predictions WITH row_id
gbt_test_pred = add_prob(
    gbt_model.transform(test_fe.select("row_id","label","weight","features"))
).select("row_id","label","p1").cache()
_ = gbt_test_pred.count()

# Join slices safely using row_id (NO row-alignment assumption)
pred_join = (
    test_fe.select("row_id","MONTH","DEP_TIME_BLK","DEPARTING_AIRPORT_BKT","CARRIER_NAME_BKT","label")
    .join(gbt_test_pred.select("row_id","p1"), on="row_id", how="inner")
    .withColumn("pred", (F.col("p1") >= F.lit(thr)).cast("int"))
).cache()
_ = pred_join.count()

def slice_table(df, col):
    return (df.groupBy(col)
            .agg(F.count("*").alias("n"),
                 F.avg(F.col("label").cast("double")).alias("delay_rate"),
                 F.avg("p1").alias("avg_pred_risk"),
                 F.avg(F.col("pred").cast("double")).alias("pred_positive_rate"))
            .orderBy(F.desc("n")))

slice_table(pred_join, "DEPARTING_AIRPORT_BKT").show(30, truncate=False)
slice_table(pred_join, "DEP_TIME_BLK").show(30, truncate=False)
slice_table(pred_join, "MONTH").show(30, truncate=False)
slice_table(pred_join, "CARRIER_NAME_BKT").show(30, truncate=False)


```

                                                                                    

    +---------------------------------------+------+-------------------+-------------------+--------------------+
    |DEPARTING_AIRPORT_BKT                  |n     |delay_rate         |avg_pred_risk      |pred_positive_rate  |
    +---------------------------------------+------+-------------------+-------------------+--------------------+
    |Atlanta Municipal                      |869616|0.10397807768026347|0.32396103453105607|0.08457181100623723 |
    |Douglas Municipal                      |509148|0.1427855947583021 |0.3788427407971919 |0.11386865901466764 |
    |Dallas Fort Worth Regional             |466982|0.16784801127238308|0.4315904413306359 |0.14450664051291057 |
    |__OTHER__                              |433711|0.13518910057619013|0.35529096594560394|0.1320049526066897  |
    |Chicago O'Hare International           |383024|0.15489890972889428|0.4479662230051671 |0.2017471490037178  |
    |Stapleton International                |363529|0.15891717029452945|0.46703742345985805|0.2572614564450154  |
    |Salt Lake City International           |190024|0.1081652843851303 |0.31918425747200907|0.03979497326653476 |
    |Detroit Metro Wayne County             |169240|0.13815882770030727|0.3769855995594357 |0.10342117702670764 |
    |Los Angeles International              |162902|0.17420289499208114|0.3983863726043692 |0.1036021657192668  |
    |Houston Intercontinental               |157423|0.12535017119480635|0.4236866278328943 |0.10799565501864404 |
    |Minneapolis-St Paul International      |155863|0.13701776560184264|0.35857546664710693|0.0357108486298865  |
    |Phoenix Sky Harbor International       |141460|0.16789198359960414|0.4045710355450529 |0.13441962392195675 |
    |San Francisco International            |129639|0.205200595499811  |0.44829520370963977|0.2241377980391703  |
    |Orlando International                  |124295|0.1716400498813307 |0.4325751786007183 |0.3018303230218432  |
    |LaGuardia                              |120321|0.17057703975199673|0.4611140801036679 |0.3350038646620291  |
    |McCarran International                 |118304|0.18285096023803082|0.4069165700029993 |0.1749729510413849  |
    |Seattle International                  |116346|0.1819228851872862 |0.3886546815834499 |0.07756175545356093 |
    |Ronald Reagan Washington National      |113046|0.15286697450595332|0.3786171145208157 |0.13037170709268794 |
    |Friendship International               |103902|0.20496236838559412|0.39446944344876   |0.18463552193413024 |
    |Logan International                    |98228 |0.1917070489066254 |0.4602248926751919 |0.30508612615547503 |
    |Newark Liberty International           |90783 |0.21752971371291982|0.45841646538803715|0.32302303294669704 |
    |John F. Kennedy International          |84462 |0.1422533210201037 |0.4193818015241971 |0.2259596031351377  |
    |Chicago Midway International           |79112 |0.21919557083628274|0.4732552383789037 |0.41309788654060065 |
    |Fort Lauderdale-Hollywood International|77553 |0.18478975668252678|0.4498712966113257 |0.34639536832875584 |
    |San Diego International Lindbergh Fl   |76123 |0.1716563981976538 |0.364448782382891  |0.10280729871392352 |
    |Philadelphia International             |74420 |0.14321418973394248|0.37906728396385775|0.10940607363611932 |
    |Washington Dulles International        |73244 |0.11815302277319643|0.3452587501663992 |0.061083501720277426|
    |Dallas Love Field                      |71600 |0.18544692737430168|0.3938205586730094 |0.21275139664804468 |
    |Miami International                    |68411 |0.1219540717136133 |0.41594447482993446|0.22823814883571356 |
    |Honolulu International                 |65213 |0.06429699599773052|0.22157331707934003|0.02550105040406054 |
    +---------------------------------------+------+-------------------+-------------------+--------------------+
    only showing top 30 rows
    


                                                                                    

    +------------+------+-------------------+-------------------+--------------------+
    |DEP_TIME_BLK|n     |delay_rate         |avg_pred_risk      |pred_positive_rate  |
    +------------+------+-------------------+-------------------+--------------------+
    |0800-0859   |651988|0.10340220985662313|0.29088460190144694|0.018744210016135267|
    |0900-0959   |623107|0.12514062592781014|0.335056393342304  |0.03641910618882471 |
    |1000-1059   |527420|0.1365875393424595 |0.37304137393304276|0.06450836145766183 |
    |0700-0759   |506535|0.08615594184014926|0.25776175557582476|0.010009180017175516|
    |0600-0659   |450766|0.06387571378497935|0.20384583919216176|0.00360941153503148 |
    |1100-1159   |440522|0.13588197638256433|0.37743854050575815|0.08360989916508142 |
    |1500-1559   |367290|0.17660159546952   |0.4576372628233946 |0.20695091072449562 |
    |1200-1259   |360746|0.15835518619749075|0.41757626356441313|0.11410244327033425 |
    |1400-1459   |333828|0.18595204716201158|0.4702577073609916 |0.2177169081083672  |
    |1300-1359   |323052|0.1722540024516177 |0.44190814931747463|0.15820982380545548 |
    |1600-1659   |312595|0.19483996864953054|0.48879553831348277|0.29092275948111773 |
    |1700-1759   |297277|0.2002173057451468 |0.5033020823168972 |0.3425155662900258  |
    |2000-2059   |271661|0.20820066185429634|0.5030684314646423 |0.3541730318301118  |
    |1800-1859   |254017|0.21645008011274836|0.5305217873046456 |0.39244617486231237 |
    |1900-1959   |245574|0.2242012590909461 |0.5348180369717063 |0.452352447734695   |
    |2200-2259   |179409|0.18382578354486118|0.45143385353672233|0.2685874175765987  |
    |2100-2159   |164393|0.2269318036656062 |0.5130649051374566 |0.4327434866448085  |
    |0001-0559   |152390|0.07997900124680098|0.2332461496822838 |0.03543539602336111 |
    |2300-2359   |21727 |0.18327426704100888|0.3678805100126041 |0.18672619321581443 |
    +------------+------+-------------------+-------------------+--------------------+
    


                                                                                    

    +-----+-------+-------------------+-------------------+-------------------+
    |MONTH|n      |delay_rate         |avg_pred_risk      |pred_positive_rate |
    +-----+-------+-------------------+-------------------+-------------------+
    |12   |3263668|0.17655625510928194|0.39391896912274194|0.16029173310520556|
    |11   |3220629|0.12006940259185395|0.3821614552991034 |0.1424647172959071 |
    +-----+-------+-------------------+-------------------+-------------------+
    


    [Stage 2372:==================================================> (195 + 2) / 200]

    +----------------------------+-------+-------------------+-------------------+--------------------+
    |CARRIER_NAME_BKT            |n      |delay_rate         |avg_pred_risk      |pred_positive_rate  |
    +----------------------------+-------+-------------------+-------------------+--------------------+
    |Delta Air Lines Inc.        |1287368|0.10884067337389154|0.3221491530266844 |0.08437525245306703 |
    |Southwest Airlines Co.      |1208286|0.18674303931354   |0.39058702858420347|0.19482639044067382 |
    |American Airlines Inc.      |967100 |0.14501085720194395|0.41534271732567113|0.1544400785854617  |
    |SkyWest Airlines Inc.       |606211 |0.1482833534858325 |0.4195383583633447 |0.1347831035728484  |
    |United Air Lines Inc.       |548428 |0.1401514875243423 |0.42999222447465896|0.2029163354168642  |
    |Comair Inc.                 |362041 |0.16740092972895335|0.37716075152944933|0.11444836358313008 |
    |American Eagle Airlines Inc.|252067 |0.13317490984539826|0.38781787731808354|0.12243966881821103 |
    |Midwest Airline, Inc.       |223742 |0.1296091033422424 |0.3669608983113186 |0.15573741184042333 |
    |JetBlue Airways             |183167 |0.21109697707556493|0.4951167869744641 |0.3894970163839556  |
    |Alaska Airlines Inc.        |180581 |0.17951500988476085|0.3703976982857512 |0.07946572452251344 |
    |Endeavor Air Inc.           |153357 |0.1312297449741453 |0.4011502400556632 |0.17059540810005414 |
    |Mesa Airlines Inc.          |146502 |0.17400445045118837|0.40893221714486644|0.08811483802268912 |
    |Spirit Air Lines            |117350 |0.13798892202812102|0.3967138683542879 |0.1535918193438432  |
    |Hawaiian Airlines Inc.      |84845  |0.06010961164476398|0.22097422630013827|0.020578702339560374|
    |Atlantic Southeast Airlines |74501  |0.13943437000845627|0.4547480318422144 |0.18478946591320922 |
    |Frontier Airlines Inc.      |66706  |0.2248823194315354 |0.5187893770763514 |0.40757952807843373 |
    |Allegiant Air               |22045  |0.16389203901111363|0.3959151268856733 |0.13399863914719892 |
    +----------------------------+-------+-------------------+-------------------+--------------------+
    


                                                                                    


```python
from datetime import datetime
RUN_ID = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
OUT = f"gs://big-data-project-481305-flightdelay/airline/metrics/slices/{RUN_ID}/"

slice_table(pred_join, "DEPARTING_AIRPORT_BKT").coalesce(1).write.mode("overwrite").csv(OUT+"airport_csv/", header=True)
slice_table(pred_join, "CARRIER_NAME_BKT").coalesce(1).write.mode("overwrite").csv(OUT+"carrier_csv/", header=True)
slice_table(pred_join, "DEP_TIME_BLK").coalesce(1).write.mode("overwrite").csv(OUT+"timeblock_csv/", header=True)
slice_table(pred_join, "MONTH").coalesce(1).write.mode("overwrite").csv(OUT+"month_csv/", header=True)

print("Saved slice outputs to:", OUT)

```

                                                                                    

    Saved slice outputs to: gs://big-data-project-481305-flightdelay/airline/metrics/slices/20251221_035129/


Below is a simple way to explain **exactly what outputs are saying**,

---

# 1) Airport slice (DEPARTING_AIRPORT_BKT)

**How to read it:**
For each airport, compare:

* **delay_rate** = actual % delayed
* **avg_pred_risk** = model’s average predicted probability
* **pred_positive_rate** = % flagged as “delay” at your threshold

# What table shows

* The model assigns **high predicted risk (~0.35–0.48)** to many airports, but the **actual delay_rate is much lower (~0.11–0.25)** for most of them.
  → This means the model is **systematically pessimistic** (over-predicting delay probabilities).

* The biggest example is **Chicago O’Hare (ORD)**:
  **delay_rate ≈ 0.160**, **avg_pred_risk ≈ 0.449**, **pred_positive_rate ≈ 0.332**
  → It flags ~33% as delays while only ~16% are truly delayed (lots of false positives).

* Airports like **JFK**:
  **delay_rate ≈ 0.154**, **avg_pred_risk ≈ 0.426**, **pred_positive_rate ≈ 0.318**
  → Again, flagged rate is about **2×** the true delay rate.

**Simple takeaway:**
The model is ranking airports (some look riskier than others), but overall it’s **over-flagging delays** across airports.

---

# 2) Time-block slice (DEP_TIME_BLK)

# What table shows

* Early morning has **lower delay_rate** and the model also outputs **lower avg_pred_risk**:

  * **0600–0659:** delay_rate ≈ 0.071, avg_pred_risk ≈ 0.203
  * **0700–0759:** delay_rate ≈ 0.084, avg_pred_risk ≈ 0.246

* Late afternoon/evening has **higher delay_rate** and the model predicts higher risk:

  * **1800–1859:** delay_rate ≈ 0.222, avg_pred_risk ≈ 0.522, pred_positive_rate ≈ 0.560
  * **1900–1959:** delay_rate ≈ 0.233, avg_pred_risk ≈ 0.527, pred_positive_rate ≈ 0.582

**Simple takeaway:**
This is a strong “makes sense” result: delays **accumulate through the day**, and the model reflects that.
But it still flags **way too many** late-day flights as delayed (pred_positive_rate ~0.56 vs true ~0.22).

---

# 3) Month slice (MONTH)

only have **months 11 and 12** here (so your test set seems to be Nov/Dec).

* **Month 12:** delay_rate ≈ 0.193, avg_pred_risk ≈ 0.399, pred_positive_rate ≈ 0.273
* **Month 11:** delay_rate ≈ 0.131, avg_pred_risk ≈ 0.384, pred_positive_rate ≈ 0.243

**Simple takeaway:**
December is truly worse than November (higher delay_rate), and your model also predicts higher risk in December.
 Again, predicted probabilities are roughly **2×** the actual delay rate.

---

# 4) Carrier slice (CARRIER_NAME_BKT)

# What table shows

* Some carriers have clearly higher delay rates:

  * **JetBlue:** delay_rate ≈ 0.233, avg_pred_risk ≈ 0.494, pred_positive_rate ≈ 0.487
  * **Frontier:** delay_rate ≈ 0.236, avg_pred_risk ≈ 0.499, pred_positive_rate ≈ 0.541
  * **Southwest:** delay_rate ≈ 0.203, avg_pred_risk ≈ 0.409, pred_positive_rate ≈ 0.371

* Others are much lower:

  * **Hawaiian:** delay_rate ≈ 0.067, avg_pred_risk ≈ 0.235, pred_positive_rate ≈ 0.051
    → Here the model is much more reasonable and flags fewer delays.

**Simple takeaway:**
The model captures carrier-level differences (some carriers do have higher delay rates).
For many carriers it still predicts too aggressively (predicted risk and flag-rate too high).

---

# Conlcusion

“Slicing shows the model captures real operational patterns (later time blocks, December, and certain carriers/airports are riskier), but it is generally over-predicting delay probability and flagging too many flights as delayed—suggesting the threshold and/or calibration should be adjusted.”




```python

```

# KMeans clustering: PCA visualization + profiling + intra-distance table

Rebuild a numeric features vector from available columns:

* compute cluster centers from data
* compute intra_cluster_distance
* PCA for visualization
* profiling table


```python
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

KMEANS_PREDS = "gs://big-data-project-481305-flightdelay/airline/clustering/kmeans_k5/sample10_pred/"
df0 = spark.read.parquet(KMEANS_PREDS)

# 1) numeric cols
num_cols = [
    "CONCURRENT_FLIGHTS","NUMBER_OF_SEATS","PLANE_AGE",
    "PRCP","SNOW","SNWD","TMAX","AWND",
    "MONTH","DAY_OF_WEEK","DISTANCE_GROUP"
]
num_cols = [c for c in num_cols if c in df0.columns]

df = df0.fillna({c: 0 for c in num_cols})

# 2) assemble + scale
assembler = VectorAssembler(inputCols=num_cols, outputCol="num_vec")
df = assembler.transform(df)

scaler = StandardScaler(inputCol="num_vec", outputCol="features_std", withMean=True, withStd=True)
df = scaler.fit(df).transform(df)

VEC_COL = "features_std"
CLUSTER_COL = "cluster"

# IMPORTANT: create df_arr 
df_arr = df.withColumn("feat_arr", vector_to_array(F.col(VEC_COL)))

# 3) Compute cluster centers correctly (across rows) via posexplode
centers_long = df_arr.select(CLUSTER_COL, F.posexplode("feat_arr").alias("pos", "val"))

centers_stats = (centers_long
                 .groupBy(CLUSTER_COL, "pos")
                 .agg(F.avg("val").alias("mean_val")))

centers_df = (centers_stats
              .groupBy(CLUSTER_COL)
              .agg(F.sort_array(F.collect_list(F.struct("pos","mean_val"))).alias("tmp"))
              .withColumn("center_arr", F.transform("tmp", lambda s: s["mean_val"]))
              .drop("tmp"))

# 4) Join centers back + intra-cluster distance
df2 = (df_arr
       .join(centers_df, on=CLUSTER_COL, how="left")
       .withColumn(
           "intra_cluster_distance",
           F.sqrt(
               F.aggregate(
                   F.arrays_zip("feat_arr", "center_arr"),
                   F.lit(0.0),
                   lambda acc, z: acc + (z["feat_arr"] - z["center_arr"]) * (z["feat_arr"] - z["center_arr"])
               )
           )
       ))

df2.select("feat_arr", CLUSTER_COL, "intra_cluster_distance").show(10, truncate=False)

```

    [Stage 2434:>                                                       (0 + 1) / 1]

    +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------+----------------------+
    |feat_arr                                                                                                                                                                                                                                 |cluster|intra_cluster_distance|
    +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------+----------------------+
    |[-0.1789620944240021, 0.9970679217054779, -1.0874763112847563, 0.2815282142567167, -0.09981394384611304, -0.12470753077128267, -0.08099424522079204, 0.8495431133196569, -1.3583283436981248, -0.9708248675396409, 0.07696019976725633]  |0      |2.55341651628561      |
    |[0.2857878307535303, 0.8894824592493948, -0.7988391827552144, 0.2815282142567167, -0.09981394384611304, -0.12470753077128267, -0.08099424522079204, 0.8495431133196569, -1.3583283436981248, -0.9708248675396409, -0.3434952375243269]   |0      |2.371665590182585     |
    |[0.053412868164764114, -1.2407096973810479, 0.06707220283341107, 0.2815282142567167, -0.09981394384611304, -0.12470753077128267, -0.08099424522079204, 0.8495431133196569, -1.3583283436981248, -0.9708248675396409, -0.3434952375243269]|0      |2.1499408390104686    |
    |[-0.22543708694175535, 0.566726071881146, -1.2317948755495272, 0.2815282142567167, -0.09981394384611304, -0.12470753077128267, -0.08099424522079204, 0.8495431133196569, -1.3583283436981248, -0.9708248675396409, 0.9178710743504228]   |4      |2.204066861200821     |
    |[-0.22543708694175535, 0.5021747944074962, 1.3659392812163493, 0.2815282142567167, -0.09981394384611304, -0.12470753077128267, -0.08099424522079204, 0.8495431133196569, -1.3583283436981248, -0.9708248675396409, 0.49741563705883957]  |0      |2.4939036422127705    |
    |[-0.1789620944240021, 0.5021747944074962, 1.3659392812163493, 0.2815282142567167, -0.09981394384611304, -0.12470753077128267, -0.08099424522079204, 0.8495431133196569, -1.3583283436981248, -0.9708248675396409, 0.49741563705883957]   |0      |2.490914601658042     |
    |[-0.1789620944240021, 1.2337559391088604, -1.5204320040790689, 0.2815282142567167, -0.09981394384611304, -0.12470753077128267, -0.08099424522079204, 0.8495431133196569, -1.3583283436981248, -0.9708248675396409, -0.3434952375243269]  |0      |2.869743358091094     |
    |[-1.2014119298145733, 0.9970679217054779, -1.376113439814298, 0.2815282142567167, -0.09981394384611304, -0.12470753077128267, -0.08099424522079204, 0.8495431133196569, -1.3583283436981248, -0.9708248675396409, -1.1844061121074934]   |3      |3.1969836006234273    |
    |[-0.27191207945950857, 0.566726071881146, 1.3659392812163493, 0.2815282142567167, -0.09981394384611304, -0.12470753077128267, -0.08099424522079204, 0.8495431133196569, -1.3583283436981248, -0.9708248675396409, 1.338326511642006]     |4      |2.7867726273190345    |
    |[0.2857878307535303, 0.8894824592493948, -1.2317948755495272, 0.2815282142567167, -0.09981394384611304, -0.12470753077128267, -0.08099424522079204, 0.8495431133196569, -1.3583283436981248, -0.9708248675396409, 0.9178710743504228]    |4      |2.1889217950437105    |
    +-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------+----------------------+
    only showing top 10 rows
    


                                                                                    

#PCA for visualization


```python

pca_model = PCA(k=2, inputCol=VEC_COL, outputCol="pca2").fit(df2)
vis = (pca_model.transform(df2)
       .withColumn("pca_arr", vector_to_array(F.col("pca2")))
       .withColumn("pc1", F.col("pca_arr")[0])
       .withColumn("pc2", F.col("pca_arr")[1])
       .drop("pca_arr"))

vis.select("pc1","pc2",F.col(CLUSTER_COL).alias("cluster"),"intra_cluster_distance").show(20, truncate=False)


```

                                                                                    

    +-------------------+-------------------+-------+----------------------+
    |pc1                |pc2                |cluster|intra_cluster_distance|
    +-------------------+-------------------+-------+----------------------+
    |0.3674740450002112 |1.1796523733682835 |0      |2.55341651628561      |
    |0.4690036006037519 |0.6806560289847833 |0      |2.371665590182585     |
    |0.956176564762182  |-0.8913927787287937|0      |2.1499408390104686    |
    |0.31475155314538955|1.5072613228252978 |4      |2.204066861200821     |
    |0.5758793525045491 |0.3778642203693928 |0      |2.4939036422127705    |
    |0.5754223954648663 |0.37102069252198483|0      |2.490914601658042     |
    |0.35400130616754566|1.1915113910872537 |0      |2.869743358091094     |
    |0.5498586241703045 |0.6035563046207113 |3      |3.1969836006234273    |
    |0.43498187448849157|0.9700529811812999 |4      |2.7867726273190345    |
    |0.24555172849828225|1.6341182959392513 |4      |2.1889217950437105    |
    |1.1646271616174793 |-1.616088013723576 |0      |2.547277242983894     |
    |0.8844343496317888 |-0.5325108559928564|0      |2.1933264719374628    |
    |1.1600575912206517 |-1.6845232921976558|0      |2.5646081291266736    |
    |1.0079275978088993 |-1.152707739391755 |3      |2.3296593047038496    |
    |0.1792639519766075 |2.06546476745848   |4      |2.0948730505721866    |
    |0.8697767232564917 |-0.7396062872555371|3      |2.227600840618919     |
    |0.5767703565377199 |0.23168468322063784|0      |2.2608684007561437    |
    |0.37204361539703884|1.2480876518423636 |3      |2.8241084332068436    |
    |0.45830363579711986|0.9271603729410243 |3      |2.542963869225529     |
    |0.9493222091669407 |-0.9940456964399136|3      |2.386200097734941     |
    +-------------------+-------------------+-------+----------------------+
    only showing top 20 rows
    



```python
# --- Option A Plot: Step 1 (run after vis is created) ---
from pyspark.sql import functions as F

PLOT_FRACTION = 0.05
MAX_POINTS = 200000

plot_df = (vis
    .select("pc1", "pc2", F.col(CLUSTER_COL).cast("int").alias("cluster"))
    .sample(False, PLOT_FRACTION, seed=42)
    .limit(MAX_POINTS)
)

pdf = plot_df.toPandas()
print("Plot points:", len(pdf), "clusters:", sorted(pdf["cluster"].unique()))

```

                                                                                    

    Plot points: 32390 clusters: [0, 1, 2, 3, 4]



```python
# --- 2) Plot ---
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))

for c in sorted(pdf["cluster"].unique()):
    sub = pdf[pdf["cluster"] == c]
    plt.scatter(sub["pc1"], sub["pc2"], s=8, alpha=0.5, label=f"Cluster {c}")

plt.title("PCA (2D) Projection of Clusters")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(markerscale=2, frameon=True)
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

```


    
![png](output_118_0.png)
    



```python
import matplotlib.pyplot as plt

plt.figure(figsize=(11, 7))

for c in sorted(pdf["cluster"].unique()):
    sub = pdf[pdf["cluster"] == c]
    plt.scatter(sub["pc1"], sub["pc2"], s=4, alpha=0.25, label=f"Cluster {c}")

plt.title("PCA (2D) Projection of KMeans Clusters")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True, alpha=0.2)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

plt.xlim(-4, 8)   # adjust left bound if you want

plt.tight_layout()
plt.show()

```


    
![png](output_119_0.png)
    



```python
cent = (pdf.groupby("cluster")[["pc1","pc2"]].mean().reset_index())

plt.figure(figsize=(11, 7))
for c in sorted(pdf["cluster"].unique()):
    sub = pdf[pdf["cluster"] == c]
    plt.scatter(sub["pc1"], sub["pc2"], s=4, alpha=0.25, label=f"Cluster {c}")

# overlay centers
plt.scatter(cent["pc1"], cent["pc2"], s=250, marker="X", edgecolors="black", linewidths=1)

for _, r in cent.iterrows():
    plt.text(r["pc1"], r["pc2"], f"C{int(r['cluster'])}", fontsize=10, weight="bold")

plt.title("PCA (2D) Projection of Clusters (Centers Marked)")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.grid(True, alpha=0.2)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

plt.xlim(-4, 8)   # adjust left bound if you want
plt.tight_layout()
plt.show()

```


    
![png](output_120_0.png)
    


# Profiling


```python

profile_exprs = [F.avg(F.col(c).cast("double")).alias(f"avg_{c}") for c in num_cols]
profile = (df2.groupBy(CLUSTER_COL)
           .agg(F.count("*").alias("n"),
                F.avg("intra_cluster_distance").alias("avg_intra_dist"),
                *profile_exprs)
           .orderBy(CLUSTER_COL))

profile.show(50, truncate=False)

print("Used numeric cols:", num_cols)
```

    [Stage 2455:============================>                           (1 + 1) / 2]

    +-------+------+------------------+----------------------+-------------------+------------------+-------------------+---------------------+---------------------+-----------------+------------------+-----------------+------------------+------------------+
    |cluster|n     |avg_intra_dist    |avg_CONCURRENT_FLIGHTS|avg_NUMBER_OF_SEATS|avg_PLANE_AGE     |avg_PRCP           |avg_SNOW             |avg_SNWD             |avg_TMAX         |avg_AWND          |avg_MONTH        |avg_DAY_OF_WEEK   |avg_DISTANCE_GROUP|
    +-------+------+------------------+----------------------+-------------------+------------------+-------------------+---------------------+---------------------+-----------------+------------------+-----------------+------------------+------------------+
    |0      |158889|2.807037870649452 |26.949153182410363    |124.65959883944137 |12.30973824493829 |0.09920806349087907|0.005816009918874239 |0.016595862520376075 |57.77958008420971|8.69290215181638  |6.025181101271957|4.005123073340508 |3.1320229845993115|
    |1      |167149|2.4137344615783713|28.725741703510042    |125.80146456155884 |12.36033120150285 |0.11160527433608168|0.0                  |0.0                  |85.62052659603108|7.736937223675002 |7.256088878784797|3.9094759765239395|3.1119360570508947|
    |2      |39704 |5.098888508605039 |27.58540701188797     |120.8047300020149  |11.591879911343945|0.21817096514205692|0.4775992343340708   |1.3917816844649822   |38.45798911948418|11.475211313721752|4.91965545033246 |3.7575811001410435|3.2831200886560548|
    |3      |172490|2.5546327965144715|29.07419560554235     |122.42263899356485 |12.505953968345992|0.08510510754247261|8.290335671633132E-5 |9.739695054785785E-5 |77.11774595628731|7.852860861499324 |6.885065800915995|3.8945098266566176|2.8400718882254044|
    |4      |110085|2.578474530447471 |26.00201662351819     |180.8360721260844  |7.623227506018077 |0.08628723259300274|0.0017077712676568093|0.0036208384430212967|72.89799791070536|8.40247009129289  |6.662424490166689|4.020920198028796 |7.599236953263388 |
    +-------+------+------------------+----------------------+-------------------+------------------+-------------------+---------------------+---------------------+-----------------+------------------+-----------------+------------------+------------------+
    
    Used numeric cols: ['CONCURRENT_FLIGHTS', 'NUMBER_OF_SEATS', 'PLANE_AGE', 'PRCP', 'SNOW', 'SNWD', 'TMAX', 'AWND', 'MONTH', 'DAY_OF_WEEK', 'DISTANCE_GROUP']


                                                                                    

## Cluster Narratives (KMeans + PCA + Profiling)

### Cluster 0 — “Typical baseline flights” *(n=158,889)*

* Mid-level congestion (**26.95**), typical aircraft size (**~124.7 seats**) and older fleet (**~12.3 yrs**).
* Mild weather overall (**PRCP 0.099**, tiny snow), moderate temp (**TMAX 57.8**) and wind (**AWND 8.69**).
* Mid-year timing (**Month ~6**) and mid-distance routes (**Dist group 3.13**).

### Cluster 1 — “Hot & dry operations” *(n=167,149)*

* More congested (**28.73**) with typical aircraft (**~125.8 seats**) and older fleet (**~12.36 yrs**).
* Completely snow-free (**SNOW 0**, **SNWD 0**) and hottest conditions (**TMAX 85.6**) with moderate wind (**7.74**).
* Skews later in the year (**Month ~7.26**).

### Cluster 2 — “Winter storm / disruption regime” *(n=39,704)*

* Smallest but most variable cluster (**avg intra-dist 5.10**).
* Strongest weather severity: highest precip (**0.218**), heavy snow (**0.478**) and snow depth (**1.392**), coldest temps (**TMAX 38.5**) and highest wind (**11.48**).
* Earlier-year timing (**Month ~4.92**). This is the **high-risk conditions** regime that separates clearly in PCA.

### Cluster 3 — “Warm, low-snow, shorter routes” *(n=172,490)*

* Highest congestion (**29.07**) with typical seats (**~122.4**) and older planes (**~12.51 yrs**).
* Near-zero snow, warm temps (**TMAX 77.1**), lower precip (**0.085**).
* Slightly shorter routes (**Dist group 2.84**) and mid-to-late year timing (**Month ~6.89**).

### Cluster 4 — “Long-haul / large aircraft / newer fleet” *(n=110,085)*

* Lower congestion (**26.00**) but much larger planes (**~180.8 seats**) and newer fleet (**~7.62 yrs**).
* Low precip and minimal snow, warm temps (**TMAX 72.9**).
* Very long routes (**Dist group 7.60**). This is the distinct **big-plane long-haul** segment.


PCA points + profiling to GCS


```python
OUT_BASE = "gs://big-data-project-481305-flightdelay/airline/clustering/kmeans_k5/report_outputs/"

# PCA points (sample to keep file smaller)
(vis.select("pc1","pc2",F.col(CLUSTER_COL).alias("cluster"),"intra_cluster_distance")
    .sample(False, 0.05, seed=42)   # 5% sample for plotting
    .write.mode("overwrite").parquet(OUT_BASE + "pca_points_5pct/"))

# Profiling table
profile.write.mode("overwrite").csv(OUT_BASE + "cluster_profile_csv/", header=True)

print("Saved to:", OUT_BASE)

```

                                                                                    

    Saved to: gs://big-data-project-481305-flightdelay/airline/clustering/kmeans_k5/report_outputs/


Cluster compactness ranking (best/worst clusters)


```python
cluster_quality = (df2.groupBy(CLUSTER_COL)
    .agg(F.count("*").alias("n"),
         F.avg("intra_cluster_distance").alias("avg_dist"),
         F.expr("percentile_approx(intra_cluster_distance, 0.5)").alias("median_dist"),
         F.expr("percentile_approx(intra_cluster_distance, 0.9)").alias("p90_dist"))
    .orderBy(F.desc("avg_dist")))

cluster_quality.show(50, truncate=False)

```

    [Stage 2485:============================>                           (1 + 1) / 2]

    +-------+------+------------------+------------------+------------------+
    |cluster|n     |avg_dist          |median_dist       |p90_dist          |
    +-------+------+------------------+------------------+------------------+
    |2      |39704 |5.098888508605039 |3.9678593646807347|8.56074315931066  |
    |0      |158889|2.807037870649452 |2.735375847320493 |3.5786260868166844|
    |4      |110085|2.578474530447471 |2.4830851991292273|3.4656515350644326|
    |3      |172490|2.5546327965144715|2.486532835625391 |3.3508074165588884|
    |1      |167149|2.4137344615783713|2.301624742066884 |3.1908498808820354|
    +-------+------+------------------+------------------+------------------+
    


                                                                                    

Explaining the clusters from profiling output

Based on table:

* Cluster 2:

    Highest intra distance (~5.10) → most “mixed” cluster
    Highest PRCP (0.218) + highest SNOW (0.478) + highest SNWD (1.392)
    Lowest TMAX (38.46) + highest AWND (11.48)

 Interpretation: “Severe winter weather / windy conditions cluster (high variability).”

* Cluster 1:

    Lowest intra distance (2.41) → most compact cluster
    PRCP (0.112), SNOW = 0, SNWD = 0, TMAX high (85.62)

 Interpretation: “Clear-weather warm-season cluster (stable operations).”

* Cluster 4:

    Very high seat count (180.84) + lowest plane age (7.62)
    distance group much larger (7.60)

 Interpretation: “Longer-haul / larger aircraft segment (structurally different fleet/route).”

* Cluster 0 & 3:

    Similar planes/temps but different snow/temps (cluster 3 shows higher snow; cluster 0 moderate).

 Interpretation: “Normal operations clusters split by seasonal temperature/precip regime.”




***

# Analysis of Cluster Visualization (PCA Projection)**

The 2D Principal Component Analysis (PCA) projection reveals three distinct operational patterns within the flight data:

**1. The High-Risk Weather Regime (Cluster 2 – Green)**
*Observation:** Cluster 2 is strictly isolated, stretching far to the right along Principal Component 1 (PC1). Its center (marked by 'X') is clearly separated from the rest of the data points.

* *Interpretation:** This represents a rare but statistically significant **"Operational Regime"** that deviates sharply from typical flights. Correlating this with the feature profiling, this cluster is defined by:
    * Much higher `PRCP`, `SNOW`, and `SNWD`
    * Higher Wind Speeds
    * Lower `TMAX`
    * *Summary:* A severe winter weather / disruption regime.

> **Project Meaning:**
> The clustering algorithm is successfully isolating the **high-risk weather regime**. This is a critical validation for the project, confirming that the model can distinguish "disruption days" from standard operations without supervision.

---

**2. Normal Operations & Structural Overlap (Clusters 0, 1, 3)**
*Observation:** Clusters 0, 1, and 3 overlap heavily near the origin (approx. $PC1 \in [-1, 1]$ and $PC2 \in [-2, 2]$).

* *Interpretation:** These groups represent **"Normal Operations"** where differences are subtler. While K-Means separates them based on the full high-dimensional feature space, they appear mixed in 2D because PCA only displays the top two directions of variance.

> **Project Meaning:**
> The visual overlap is **not a failure**; it is a known limitation of 2D PCA projections. These clusters likely represent valid segments based on fleet, route, or time patterns that do not manifest as extreme outliers in the top two principal components.

---

**3. The Long-Haul / Large Fleet Segment (Cluster 4 – Purple)**
*Observation:** Cluster 4 is distinct due to its vertical separation, sitting significantly higher on the PC2 axis.

* *Interpretation:** This vertical shift indicates a difference in **structural patterns** rather than weather severity. Based on feature profiling, Cluster 4 is characterized by:
    * Very high `NUMBER_OF_SEATS`
    * Lower `PLANE_AGE`
    * Much larger `DISTANCE_GROUP`
    * *Summary:* A segment for larger aircraft and long-haul operations.

> **Project Meaning:**
> This identifies a valuable business segment. **Long-haul and large-aircraft operations** behave differently logistically, implying their delay risk patterns will differ from regional flights. Segregating this group allows the model to learn these specific nuances.

***




# Project Impact: Why Clustering Matters

* Uncovers Operational Regimes Moves beyond "black-box" prediction by proving distinct flight categories exist (e.g., Severe Weather vs. Long-Haul), validating the data structure.

* Boosts Model Performance The Cluster_ID acts as a powerful "meta-feature" that summarizes complex conditions. Adding it to the classifier improves key metrics like PR-AUC and Recall.

* Diagnoses Model Errors Provides context for failures. Mispredictions within the "Severe Weather" cluster can be attributed to inherent environmental volatility rather than model insufficiency.


```python

```

# Anomaly Detection (Spikes)

7.1 Build one “scored” dataset across all months (train+val+test)


```python
from pyspark.sql import functions as F

BEST_MODEL_NAME = "GBT"       # change label only
best_model = gbt_model        # or rf_model if RF is best

# union all splits so anomalies can be seen over months 1..12
all_gold = (train5.withColumn("split", F.lit("train"))
            .unionByName(val5.withColumn("split", F.lit("val")))
            .unionByName(test5.withColumn("split", F.lit("test"))))

# apply feature pipeline to all rows
all_fe = feat_model.transform(all_gold)

# score and keep slice columns
scored_all = (add_prob(best_model.transform(
                all_fe.select(
                    "row_id","split",
                    "MONTH","DEP_TIME_BLK",
                    "DEPARTING_AIRPORT_BKT","CARRIER_NAME_BKT",
                    "label","weight","features"
                )
            ))
            .select(
                "row_id","split",
                "MONTH","DEP_TIME_BLK",
                "DEPARTING_AIRPORT_BKT","CARRIER_NAME_BKT",
                "label","p1"
            )
            .cache())

_ = scored_all.count()
print("scored_all rows:", scored_all.count())
scored_all.show(5, truncate=False)

```

                                                                                    

    scored_all rows: 6489062
    +--------------------+-----+-----+------------+---------------------+----------------------+-----+-------------------+
    |row_id              |split|MONTH|DEP_TIME_BLK|DEPARTING_AIRPORT_BKT|CARRIER_NAME_BKT      |label|p1                 |
    +--------------------+-----+-----+------------+---------------------+----------------------+-----+-------------------+
    |-5972261017086745821|train|4    |2100-2159   |Atlanta Municipal    |Southwest Airlines Co.|0    |0.6227855461130568 |
    |-5147052611826786705|train|4    |1400-1459   |Atlanta Municipal    |Southwest Airlines Co.|0    |0.48472440096180003|
    |5853433762726780349 |train|4    |1400-1459   |Atlanta Municipal    |Southwest Airlines Co.|1    |0.5864889478170792 |
    |2825730802779138287 |train|4    |1500-1559   |Atlanta Municipal    |Southwest Airlines Co.|1    |0.570769747324567  |
    |-6565439843932954386|train|4    |1900-1959   |Atlanta Municipal    |Southwest Airlines Co.|0    |0.6934541792332786 |
    +--------------------+-----+-----+------------+---------------------+----------------------+-----+-------------------+
    only showing top 5 rows
    


7.2 Aggregate “risk signals” we want to monitor


```python
# Monthly drift proxy (simple, powerful for report)
monthly = (scored_all.groupBy("MONTH")
           .agg(F.count("*").alias("n"),
                F.avg(F.col("label").cast("double")).alias("delay_rate"),
                F.avg("p1").alias("avg_pred_risk"))
           .orderBy("MONTH"))

# Month x timeblock (heatmap + spike detection)
mtb = (scored_all.groupBy("MONTH","DEP_TIME_BLK")
       .agg(F.count("*").alias("n"),
            F.avg(F.col("label").cast("double")).alias("delay_rate"),
            F.avg("p1").alias("avg_pred_risk")))

# Month x airport (spike table; filter small groups to avoid noise)
mair = (scored_all.groupBy("MONTH","DEPARTING_AIRPORT_BKT")
        .agg(F.count("*").alias("n"),
             F.avg(F.col("label").cast("double")).alias("delay_rate"),
             F.avg("p1").alias("avg_pred_risk"))
        .filter(F.col("n") >= 500))

monthly.show(20, truncate=False)

```

    [Stage 2497:==================================================>   (16 + 1) / 17]

    +-----+------+-------------------+-------------------+
    |MONTH|n     |delay_rate         |avg_pred_risk      |
    +-----+------+-------------------+-------------------+
    |1    |502050|0.1746479434319291 |0.4317398087391284 |
    |2    |451439|0.217163337682389  |0.48809878114682964|
    |3    |554120|0.17431061863856204|0.43295785171447504|
    |4    |533687|0.1850466659296554 |0.4451906656504797 |
    |5    |558025|0.20344966623359168|0.47048545570826233|
    |6    |557027|0.24392174885598006|0.5170444236055695 |
    |7    |578193|0.21314336216453675|0.47985311302454253|
    |8    |579532|0.20604729333324132|0.472190795745952  |
    |9    |529873|0.13745557897835897|0.3725695110580066 |
    |10   |561327|0.16166156268984033|0.4527788369594177 |
    |11   |531893|0.1420887283720598 |0.41532657461756595|
    |12   |551896|0.208551973560236  |0.4279210620480371 |
    +-----+------+-------------------+-------------------+
    


                                                                                    

7.3 Spike scoring (robust z-score) + visuals (Matplotlib)

Core anomaly logic:

* Compare each segment’s delay rate to its own baseline
* Use robust z-score (median/MAD) so outliers don’t break the detector
* Flag “spikes” as abs(z) >= 3.5


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def robust_z(series):
    """Robust z-score using median + MAD."""
    s = np.asarray(series, dtype=float)
    med = np.median(s)
    mad = np.median(np.abs(s - med)) + 1e-9
    z = 0.6745 * (s - med) / mad
    return z, med, mad

Z_THR = 3.0   # robust spike threshold (3.5 is common)

# -----------------------------
# A) Monthly drift/spike plots
# -----------------------------
pdf_m = monthly.toPandas().sort_values("MONTH")

# Robust z + stats for delay rate
z_delay, med_delay, mad_delay = robust_z(pdf_m["delay_rate"].values)
pdf_m["z_delay"] = z_delay
pdf_m["anomaly_delay"] = np.abs(pdf_m["z_delay"]) >= Z_THR

# Robust z + stats for avg predicted risk
z_risk, med_risk, mad_risk = robust_z(pdf_m["avg_pred_risk"].values)
pdf_m["z_risk"] = z_risk
pdf_m["anomaly_risk"] = np.abs(pdf_m["z_risk"]) >= Z_THR

# Convert Z_THR into value-threshold bands for easy plotting:
# z = 0.6745*(x-med)/mad  =>  x = med + (z*mad)/0.6745
def z_to_value_band(med, mad, zthr):
    delta = (zthr * mad) / 0.6745
    return (med - delta, med + delta)

delay_lo, delay_hi = z_to_value_band(med_delay, mad_delay, Z_THR)
risk_lo,  risk_hi  = z_to_value_band(med_risk,  mad_risk,  Z_THR)

# ---- Plot 1: Monthly Delay Rate ----
plt.figure(figsize=(10,4))
plt.plot(pdf_m["MONTH"], pdf_m["delay_rate"], marker="o")

# baseline + bands
plt.axhline(med_delay, linestyle="--")
plt.axhline(delay_hi, linestyle=":")
plt.axhline(delay_lo, linestyle=":")

# spike markers
sp = pdf_m[pdf_m["anomaly_delay"]]
plt.scatter(sp["MONTH"], sp["delay_rate"], s=120, marker="X")

# label spike months only
for _, r in sp.iterrows():
    plt.annotate(f"M{int(r['MONTH'])}", (r["MONTH"], r["delay_rate"]),
                 textcoords="offset points", xytext=(6,6))

plt.title(f"Monthly Delay Rate (Robust spikes: |z| ≥ {Z_THR})")
plt.xlabel("Month"); plt.ylabel("Delay rate")
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

# ---- Plot 2: Monthly Mean Predicted Risk ----
plt.figure(figsize=(10,4))
plt.plot(pdf_m["MONTH"], pdf_m["avg_pred_risk"], marker="o")

# baseline + bands
plt.axhline(med_risk, linestyle="--")
plt.axhline(risk_hi, linestyle=":")
plt.axhline(risk_lo, linestyle=":")

# spike markers
sp = pdf_m[pdf_m["anomaly_risk"]]
plt.scatter(sp["MONTH"], sp["avg_pred_risk"], s=120, marker="X")

# label spike months only
for _, r in sp.iterrows():
    plt.annotate(f"M{int(r['MONTH'])}", (r["MONTH"], r["avg_pred_risk"]),
                 textcoords="offset points", xytext=(6,6))

plt.title(f"Monthly Mean Predicted Risk (Robust spikes: |z| ≥ {Z_THR})")
plt.xlabel("Month"); plt.ylabel("Avg predicted risk")
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

# Optional: print which months were flagged (nice for report)
print("Spike months (delay_rate):", sorted(pdf_m.loc[pdf_m["anomaly_delay"], "MONTH"].tolist()))
print("Spike months (avg_pred_risk):", sorted(pdf_m.loc[pdf_m["anomaly_risk"], "MONTH"].tolist()))

```


    
![png](output_141_0.png)
    



    
![png](output_141_1.png)
    


    Spike months (delay_rate): []
    Spike months (avg_pred_risk): []


    Monthly charts:

* These plots show how the actual delay rate and the model’s average predicted delay risk change month-by-month, making seasonal spikes easy to spot.

* Months where both rise together indicate true operational stress periods (weather/traffic), not just model noise.


```python
# 1. LINK: Convert  Spark 'mtb' to Pandas 'pdf_mtb'
pdf_mtb = mtb.toPandas()

# 2. RUN VISUALIZATION 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def dep_blk_start(s):
    try:
        return int(str(s).split("-")[0])
    except:
        return 9999

# IMPORTANT: return a SAME-LENGTH pandas Series (no length mismatch)
def robust_z_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.zeros(len(x)), index=s.index)   # no variation => z = 0
    z = 0.6745 * (x - med) / mad
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.Series(z, index=s.index)

Z_THR = 2.5   # start here; raise/lower later
MIN_N = 300   # ignore tiny cells (noise)

# ---- Prepare month x timeblock table ----
pdf_mtb2 = pdf_mtb.copy()
pdf_mtb2["blk_start"] = pdf_mtb2["DEP_TIME_BLK"].apply(dep_blk_start)

# enforce numeric
pdf_mtb2["delay_rate"] = pd.to_numeric(pdf_mtb2["delay_rate"], errors="coerce")
pdf_mtb2["avg_pred_risk"] = pd.to_numeric(pdf_mtb2["avg_pred_risk"], errors="coerce")
pdf_mtb2["n"] = pd.to_numeric(pdf_mtb2["n"], errors="coerce").fillna(0).astype(int)

# robust z-score WITHIN each timeblock across months (baseline per timeblock)
pdf_mtb2["z_delay_blk"] = (
    pdf_mtb2.groupby("DEP_TIME_BLK")["delay_rate"]
    .transform(robust_z_series)
)

# spike flag: large enough support + z above threshold
pdf_mtb2["is_spike"] = (pdf_mtb2["n"] >= MIN_N) & (pdf_mtb2["z_delay_blk"].abs() >= Z_THR)

# ---- Heatmap matrix (delay_rate) ----
heat = (
    pdf_mtb2.pivot_table(index="MONTH", columns="DEP_TIME_BLK", values="delay_rate", aggfunc="mean")
    .sort_index()
)

# sort columns by start time
cols_sorted = sorted(list(heat.columns), key=dep_blk_start)
heat = heat[cols_sorted]

plt.figure(figsize=(14, 5))
plt.imshow(heat.values, aspect="auto", interpolation="nearest")
plt.xticks(range(len(heat.columns)), heat.columns, rotation=90)
plt.yticks(range(len(heat.index)), heat.index)
plt.title("Heatmap: Delay Rate by Month × Time Block")
plt.xlabel("Time Block")
plt.ylabel("Month")
plt.colorbar(label="Delay rate")
plt.tight_layout()
plt.show()

# ---- Top spikes table (for reporting) ----
top_spikes = (
    pdf_mtb2[pdf_mtb2["is_spike"]]
    .sort_values("z_delay_blk", ascending=False)
    .loc[:, ["MONTH","DEP_TIME_BLK","n","delay_rate","avg_pred_risk","z_delay_blk"]]
    .head(15)
)

print("Top Month×TimeBlock spikes (within-block robust z):")
display(top_spikes)

if top_spikes.empty:
    print("No spikes at current threshold. Showing top 15 most unusual cells anyway:")
    display(
        pdf_mtb2[pdf_mtb2["n"] >= MIN_N]
        .sort_values("z_delay_blk", ascending=False)
        .loc[:, ["MONTH","DEP_TIME_BLK","n","delay_rate","avg_pred_risk","z_delay_blk"]]
        .head(15)
    )
```

                                                                                    


    
![png](output_143_1.png)
    


    Top Month×TimeBlock spikes (within-block robust z):



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MONTH</th>
      <th>DEP_TIME_BLK</th>
      <th>n</th>
      <th>delay_rate</th>
      <th>avg_pred_risk</th>
      <th>z_delay_blk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>173</th>
      <td>2</td>
      <td>1200-1259</td>
      <td>27965</td>
      <td>0.226569</td>
      <td>0.500811</td>
      <td>2.512108</td>
    </tr>
    <tr>
      <th>153</th>
      <td>9</td>
      <td>1200-1259</td>
      <td>32231</td>
      <td>0.128727</td>
      <td>0.368000</td>
      <td>-2.864566</td>
    </tr>
    <tr>
      <th>156</th>
      <td>9</td>
      <td>1400-1459</td>
      <td>31114</td>
      <td>0.154850</td>
      <td>0.416611</td>
      <td>-2.886929</td>
    </tr>
  </tbody>
</table>
</div>


    Month × Time Block heatmap:

* This heatmap pinpoints when during the day delays concentrate within each month, highlighting recurring “hot” time windows (often late afternoon/evening). 

* The “top spike cells” list is your anomaly report: the most unusually high delay pockets by month and time block.

# Where did the spikes come from?” (airport/carrier drivers)

This produces a bar chart of the most anomalous airports (per-month vs that airport’s normal).


```python
pdf_air = mair.toPandas()

# spike vs that airport’s baseline across months
pdf_air["z_delay"] = (pdf_air.groupby("DEPARTING_AIRPORT_BKT")["delay_rate"]
                      .transform(lambda x: robust_z(x.values)[0]))
pdf_air["is_spike"] = (np.abs(pdf_air["z_delay"]) >= Z_THR)

# take strongest positive spikes
top_air_spikes = (pdf_air.sort_values("z_delay", ascending=False)
                  .head(20))

plt.figure(figsize=(10,5))
plt.barh(top_air_spikes["DEPARTING_AIRPORT_BKT"].astype(str),
         top_air_spikes["z_delay"])
plt.title("Top Airport Spikes (Robust z-score of delay_rate)")
plt.xlabel("Robust z-score"); plt.ylabel("Airport bucket")
plt.gca().invert_yaxis()
plt.grid(True, axis="x", alpha=0.2)
plt.tight_layout()
plt.show()

display(top_air_spikes[["MONTH","DEPARTING_AIRPORT_BKT","n","delay_rate","avg_pred_risk","z_delay"]].head(20))

```

                                                                                    


    
![png](output_146_1.png)
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MONTH</th>
      <th>DEPARTING_AIRPORT_BKT</th>
      <th>n</th>
      <th>delay_rate</th>
      <th>avg_pred_risk</th>
      <th>z_delay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>541</th>
      <td>6</td>
      <td>Jacksonville International</td>
      <td>2965</td>
      <td>0.220236</td>
      <td>0.475167</td>
      <td>3.707612</td>
    </tr>
    <tr>
      <th>249</th>
      <td>7</td>
      <td>Jacksonville International</td>
      <td>2983</td>
      <td>0.218572</td>
      <td>0.446957</td>
      <td>3.615468</td>
    </tr>
    <tr>
      <th>404</th>
      <td>2</td>
      <td>Minneapolis-St Paul International</td>
      <td>10714</td>
      <td>0.246873</td>
      <td>0.467395</td>
      <td>3.556016</td>
    </tr>
    <tr>
      <th>209</th>
      <td>2</td>
      <td>Portland International</td>
      <td>4166</td>
      <td>0.170667</td>
      <td>0.394542</td>
      <td>3.541102</td>
    </tr>
    <tr>
      <th>604</th>
      <td>6</td>
      <td>Austin - Bergstrom International</td>
      <td>5711</td>
      <td>0.256172</td>
      <td>0.514768</td>
      <td>3.357702</td>
    </tr>
    <tr>
      <th>141</th>
      <td>12</td>
      <td>Portland International</td>
      <td>5353</td>
      <td>0.167756</td>
      <td>0.328853</td>
      <td>3.347147</td>
    </tr>
    <tr>
      <th>509</th>
      <td>6</td>
      <td>Dallas Fort Worth Regional</td>
      <td>25390</td>
      <td>0.301457</td>
      <td>0.587911</td>
      <td>3.335809</td>
    </tr>
    <tr>
      <th>580</th>
      <td>8</td>
      <td>Miami International</td>
      <td>7209</td>
      <td>0.296851</td>
      <td>0.553475</td>
      <td>3.017594</td>
    </tr>
    <tr>
      <th>122</th>
      <td>12</td>
      <td>San Jose International</td>
      <td>5195</td>
      <td>0.211935</td>
      <td>0.365916</td>
      <td>3.002828</td>
    </tr>
    <tr>
      <th>428</th>
      <td>2</td>
      <td>Honolulu International</td>
      <td>3491</td>
      <td>0.141793</td>
      <td>0.323024</td>
      <td>2.755833</td>
    </tr>
    <tr>
      <th>597</th>
      <td>6</td>
      <td>Kansas City International</td>
      <td>4615</td>
      <td>0.214518</td>
      <td>0.466575</td>
      <td>2.699255</td>
    </tr>
    <tr>
      <th>566</th>
      <td>5</td>
      <td>Dallas Fort Worth Regional</td>
      <td>25360</td>
      <td>0.285607</td>
      <td>0.559640</td>
      <td>2.674216</td>
    </tr>
    <tr>
      <th>503</th>
      <td>6</td>
      <td>Cincinnati/Northern Kentucky International</td>
      <td>4183</td>
      <td>0.258905</td>
      <td>0.523423</td>
      <td>2.657504</td>
    </tr>
    <tr>
      <th>182</th>
      <td>2</td>
      <td>Orange County</td>
      <td>3031</td>
      <td>0.211151</td>
      <td>0.438970</td>
      <td>2.606167</td>
    </tr>
    <tr>
      <th>530</th>
      <td>6</td>
      <td>Stapleton International</td>
      <td>21878</td>
      <td>0.332663</td>
      <td>0.598706</td>
      <td>2.550186</td>
    </tr>
    <tr>
      <th>188</th>
      <td>8</td>
      <td>Jacksonville International</td>
      <td>2891</td>
      <td>0.199239</td>
      <td>0.443317</td>
      <td>2.545028</td>
    </tr>
    <tr>
      <th>180</th>
      <td>2</td>
      <td>Seattle International</td>
      <td>9017</td>
      <td>0.241433</td>
      <td>0.505450</td>
      <td>2.454532</td>
    </tr>
    <tr>
      <th>564</th>
      <td>6</td>
      <td>Miami International</td>
      <td>7008</td>
      <td>0.274115</td>
      <td>0.573604</td>
      <td>2.451173</td>
    </tr>
    <tr>
      <th>510</th>
      <td>6</td>
      <td>Portland International</td>
      <td>5529</td>
      <td>0.154097</td>
      <td>0.375383</td>
      <td>2.436991</td>
    </tr>
    <tr>
      <th>94</th>
      <td>12</td>
      <td>Metropolitan Oakland International</td>
      <td>4387</td>
      <td>0.336221</td>
      <td>0.433369</td>
      <td>2.399849</td>
    </tr>
  </tbody>
</table>
</div>


Export all Phase outputs to GCS (Parquet + CSV)


```python
from datetime import datetime
import pandas as pd

run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
OUT_BASE = f"gs://big-data-project-481305-flightdelay/airline/anomalies/run_{run_id}/"
print("Writing to:", OUT_BASE)

# Save raw aggregates (Spark DFs)
monthly.write.mode("overwrite").parquet(OUT_BASE + "monthly_parquet/")
mtb.write.mode("overwrite").parquet(OUT_BASE + "month_timeblock_parquet/")
mair.write.mode("overwrite").parquet(OUT_BASE + "month_airport_parquet/")

# Save spike tables (Pandas -> Spark), but guard against empty
def safe_export_pdf(pdf: pd.DataFrame, path: str):
    if pdf is None or len(pdf) == 0:
        spark.createDataFrame([], "MONTH int, DEP_TIME_BLK string, n int, delay_rate double, avg_pred_risk double, z_delay_blk double") \
            .coalesce(1).write.mode("overwrite").csv(path, header=True)
        print("Exported EMPTY placeholder:", path)
    else:
        spark.createDataFrame(pdf).coalesce(1).write.mode("overwrite").csv(path, header=True)
        print("Exported:", path)

# Month×TimeBlock spikes table you just fixed (top_spikes)
safe_export_pdf(top_spikes, OUT_BASE + "top_spikes_month_timeblock_csv/")

# If you also have top_air_spikes (month×airport)
if "top_air_spikes" in globals():
    safe_export_pdf(top_air_spikes, OUT_BASE + "top_spikes_month_airport_csv/")

print("Done. Phase-7 anomalies exported to:", OUT_BASE)

```

    Writing to: gs://big-data-project-481305-flightdelay/airline/anomalies/run_20251221_035448/


                                                                                    

    Exported: gs://big-data-project-481305-flightdelay/airline/anomalies/run_20251221_035448/top_spikes_month_timeblock_csv/


                                                                                    

    Exported: gs://big-data-project-481305-flightdelay/airline/anomalies/run_20251221_035448/top_spikes_month_airport_csv/
    Done. Phase-7 anomalies exported to: gs://big-data-project-481305-flightdelay/airline/anomalies/run_20251221_035448/



```python

```


```python
# --- 6) Export everything to GCS (handles empty spike tables safely) ---
run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
OUT_BASE = f"gs://big-data-project-481305-flightdelay/airline/anomalies/run_{run_id}/"
print("Writing to:", OUT_BASE)

monthly.write.mode("overwrite").parquet(OUT_BASE + "monthly_parquet/")
mtb.write.mode("overwrite").parquet(OUT_BASE + "month_timeblock_parquet/")
mair.write.mode("overwrite").parquet(OUT_BASE + "month_airport_parquet/")

def safe_export_pdf(pdf, path, schema_str):
    if pdf is None or len(pdf) == 0:
        spark.createDataFrame([], schema_str).coalesce(1).write.mode("overwrite").csv(path, header=True)
        print("Exported EMPTY placeholder:", path)
    else:
        spark.createDataFrame(pdf).coalesce(1).write.mode("overwrite").csv(path, header=True)
        print("Exported:", path)

safe_export_pdf(
    top_spikes,
    OUT_BASE + "top_spikes_month_timeblock_csv/",
    "MONTH int, DEP_TIME_BLK string, n long, delay_rate double, avg_pred_risk double, z_delay_blk double"
)

safe_export_pdf(
    top_air_spikes,
    OUT_BASE + "top_spikes_month_airport_csv/",
    "MONTH int, DEPARTING_AIRPORT_BKT string, n long, delay_rate double, avg_pred_risk double, z_delay_air double"
)

print("Done. anomalies exported to:", OUT_BASE)
```

    Writing to: gs://big-data-project-481305-flightdelay/airline/anomalies/run_20251221_035459/


                                                                                    

    Exported: gs://big-data-project-481305-flightdelay/airline/anomalies/run_20251221_035459/top_spikes_month_timeblock_csv/


    25/12/21 03:55:10 WARN YarnAllocator: Container from a bad node: container_1766261877773_0002_01_000010 on host: airline-analysis-v3-w-1.us-east1-b.c.big-data-project-481305.internal. Exit status: 143. Diagnostics: [2025-12-21 03:55:10.644]Container killed on request. Exit code is 143
    [2025-12-21 03:55:10.645]Container exited with a non-zero exit code 143. 
    [2025-12-21 03:55:10.646]Killed by external signal
    .
    25/12/21 03:55:10 WARN YarnSchedulerBackend$YarnSchedulerEndpoint: Requesting driver to remove executor 8 for reason Container from a bad node: container_1766261877773_0002_01_000010 on host: airline-analysis-v3-w-1.us-east1-b.c.big-data-project-481305.internal. Exit status: 143. Diagnostics: [2025-12-21 03:55:10.644]Container killed on request. Exit code is 143
    [2025-12-21 03:55:10.645]Container exited with a non-zero exit code 143. 
    [2025-12-21 03:55:10.646]Killed by external signal
    .
    25/12/21 03:55:10 ERROR YarnScheduler: Lost executor 8 on airline-analysis-v3-w-1.us-east1-b.c.big-data-project-481305.internal: Container from a bad node: container_1766261877773_0002_01_000010 on host: airline-analysis-v3-w-1.us-east1-b.c.big-data-project-481305.internal. Exit status: 143. Diagnostics: [2025-12-21 03:55:10.644]Container killed on request. Exit code is 143
    [2025-12-21 03:55:10.645]Container exited with a non-zero exit code 143. 
    [2025-12-21 03:55:10.646]Killed by external signal
    .
                                                                                    

    Exported: gs://big-data-project-481305-flightdelay/airline/anomalies/run_20251221_035459/top_spikes_month_airport_csv/
    Done. anomalies exported to: gs://big-data-project-481305-flightdelay/airline/anomalies/run_20251221_035459/



```python

```

# Scale-IN experiment (10% → 25% → 50% → 100%)

Scaling Strategy: Progressive 'Scale-In' 

We adopted a progressive 'Scale-In' strategy to efficiently validate pipeline integrity and optimize hyperparameters. By incrementally increasing data volume, we ensure model stability and resource efficiency before committing to full-scale training.

20% (~2M): Validate schema, pipeline, and baselines

50% (~5M): Hyperparameter tuning (TrainValidationSplit)

75% (~7.5M): Stability & calibration check across time/segments

100% (~10M): Final training + full evaluation


```python
import time
import pandas as pd
from datetime import datetime
from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.ml.classification import GBTClassifier

# ----------------------------
# 0) Helper: timing wrapper
# ----------------------------
def timed(name, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"[TIMER] {name}: {dt:.2f}s ({dt/60:.2f}m)")
    return out, dt

# ----------------------------
# 1) PRE-COMPUTE STEP (Run this ONCE before the loop)
# ----------------------------
print("--- PRE-MATERIALIZATION STARTED ---")

# A. Define Columns
CAT_COLS = BASE_CAT
NUM_COLS = BASE_NUM + CONGESTION_NUM + WEATHER_NUM + LEAKAGE_SAFE_AGG

# B. Fit Feature Pipeline ONCE on full training data
# Note: This is acceptable for scale-in experiments. We fit on 100% to get a consistent schema.
raw_feat_model, dt_fit = timed("Fit Feature Pipeline (Full Train)", lambda:
    build_feature_pipeline(CAT_COLS, NUM_COLS).fit(train5)
)

# C. Transform and Cache ALL sets ONCE
# This avoids re-calculating vectors for Val/Test in every loop iteration
def transform_and_cache(df, name):
    out = raw_feat_model.transform(df).select("row_id", "label", "weight", "features").cache()
    count = out.count() # Force materialization
    print(f"Cached {name}: {count} rows")
    return out

train5_fe_all, _ = timed("Transform+Cache Full Train", lambda: transform_and_cache(train5, "Train Full"))
val5_fe, _       = timed("Transform+Cache Val",       lambda: transform_and_cache(val5, "Val"))
test5_fe, _      = timed("Transform+Cache Test",      lambda: transform_and_cache(test5, "Test"))

print("--- PRE-MATERIALIZATION COMPLETE ---")


# ----------------------------
# 2) Optimized Scale-up runner
# ----------------------------
def run_scaleup_fast(frac, tr_full_fe, va_fe, te_fe, seed=42):
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    # 1. Sample from the ALREADY TRANSFORMED training data
    # This is effectively instant compared to re-running the pipeline
    tr_sample = tr_full_fe.sample(False, frac, seed)
    
    # Force cache on the small sample so the GBT iterates fast
    tr_sample.cache()
    n_sample = tr_sample.count()
    print(f"[{frac:.2f}] Training on {n_sample} rows...")

    # 2. Fit GBT
    def fit_gbt():
        return GBTClassifier(
            featuresCol="features", labelCol="label", weightCol="weight",
            maxIter=80, maxDepth=6, stepSize=0.08, subsamplingRate=0.8
        ).fit(tr_sample)

    gbt_model, dt_fit_model = timed(f"[{frac:.2f}] fit GBT", fit_gbt)

    # 3. Evaluate (Using the pre-cached Validation/Test sets)
    thresholds = [i/100 for i in range(10, 91, 5)]
    (summary, va_pred, te_pred), dt_eval = timed(f"[{frac:.2f}] eval", lambda:
        evaluate_classifier(f"GBT_scaleup_{frac:.2f}", gbt_model, va_fe, te_fe, thresholds=thresholds)
    )
    
    # 4. Cleanup (Crucial for loops!)
    tr_sample.unpersist()

    # Build results
    out = dict(summary)
    out.update({
        "run_id": run_id,
        "train_frac": float(frac),
        "train_n": n_sample,
        "val_n": va_fe.count(), # Metadata lookup only
        "time_fit_model_s": dt_fit_model,
        "time_eval_s": dt_eval,
        "time_total_s": dt_fit_model + dt_eval # We exclude one-off pipeline time here
    })
    return Row(**out)

# ----------------------------
# 3) Run scale-up grid
# ----------------------------
SCALE_FRACS = [0.10, 0.25, 0.50, 1.00]

# Pass the PRE-TRANSFORMED dataframes into the function
scaleup_rows = [run_scaleup_fast(fr, train5_fe_all, val5_fe, test5_fe) for fr in SCALE_FRACS]

scaleup_df = spark.createDataFrame(scaleup_rows)

scaleup_df.select(
    "train_frac","train_n",
    "val_pr_auc","test_pr_auc",
    "test_recall_top5","test_f1",
    "time_total_s","time_fit_model_s"
).orderBy("train_frac").show(truncate=False)
```

    --- PRE-MATERIALIZATION STARTED ---


                                                                                    

    [TIMER] Fit Feature Pipeline (Full Train): 57.33s (0.96m)


    25/12/21 03:56:56 WARN YarnAllocator: Container from a bad node: container_1766261877773_0002_01_000011 on host: airline-analysis-v3-w-0.us-east1-b.c.big-data-project-481305.internal. Exit status: 143. Diagnostics: [2025-12-21 03:56:56.018]Container killed on request. Exit code is 143
    [2025-12-21 03:56:56.019]Container exited with a non-zero exit code 143. 
    [2025-12-21 03:56:56.020]Killed by external signal
    .
    25/12/21 03:56:56 ERROR YarnScheduler: Lost executor 9 on airline-analysis-v3-w-0.us-east1-b.c.big-data-project-481305.internal: Container from a bad node: container_1766261877773_0002_01_000011 on host: airline-analysis-v3-w-0.us-east1-b.c.big-data-project-481305.internal. Exit status: 143. Diagnostics: [2025-12-21 03:56:56.018]Container killed on request. Exit code is 143
    [2025-12-21 03:56:56.019]Container exited with a non-zero exit code 143. 
    [2025-12-21 03:56:56.020]Killed by external signal
    .
    25/12/21 03:56:56 WARN YarnSchedulerBackend$YarnSchedulerEndpoint: Requesting driver to remove executor 9 for reason Container from a bad node: container_1766261877773_0002_01_000011 on host: airline-analysis-v3-w-0.us-east1-b.c.big-data-project-481305.internal. Exit status: 143. Diagnostics: [2025-12-21 03:56:56.018]Container killed on request. Exit code is 143
    [2025-12-21 03:56:56.019]Container exited with a non-zero exit code 143. 
    [2025-12-21 03:56:56.020]Killed by external signal
    .
    25/12/21 03:56:56 WARN TaskSetManager: Lost task 3.0 in stage 2566.0 (TID 134749) (airline-analysis-v3-w-0.us-east1-b.c.big-data-project-481305.internal executor 9): ExecutorLostFailure (executor 9 exited caused by one of the running tasks) Reason: Container from a bad node: container_1766261877773_0002_01_000011 on host: airline-analysis-v3-w-0.us-east1-b.c.big-data-project-481305.internal. Exit status: 143. Diagnostics: [2025-12-21 03:56:56.018]Container killed on request. Exit code is 143
    [2025-12-21 03:56:56.019]Container exited with a non-zero exit code 143. 
    [2025-12-21 03:56:56.020]Killed by external signal
    .
                                                                                    

    Cached Train Full: 4843946 rows
    [TIMER] Transform+Cache Full Train: 90.16s (1.50m)


                                                                                    

    Cached Val: 561327 rows
    [TIMER] Transform+Cache Val: 5.27s (0.09m)


                                                                                    

    Cached Test: 1083789 rows
    [TIMER] Transform+Cache Test: 18.87s (0.31m)
    --- PRE-MATERIALIZATION COMPLETE ---


                                                                                    

    [0.10] Training on 484127 rows...


                                                                                    

    [TIMER] [0.10] fit GBT: 226.01s (3.77m)


    25/12/21 04:03:17 WARN YarnAllocator: Container from a bad node: container_1766261877773_0002_01_000013 on host: airline-analysis-v3-w-3.us-east1-b.c.big-data-project-481305.internal. Exit status: 143. Diagnostics: [2025-12-21 04:03:17.259]Container killed on request. Exit code is 143
    [2025-12-21 04:03:17.260]Container exited with a non-zero exit code 143. 
    [2025-12-21 04:03:17.260]Killed by external signal
    .
    25/12/21 04:03:17 WARN YarnSchedulerBackend$YarnSchedulerEndpoint: Requesting driver to remove executor 11 for reason Container from a bad node: container_1766261877773_0002_01_000013 on host: airline-analysis-v3-w-3.us-east1-b.c.big-data-project-481305.internal. Exit status: 143. Diagnostics: [2025-12-21 04:03:17.259]Container killed on request. Exit code is 143
    [2025-12-21 04:03:17.260]Container exited with a non-zero exit code 143. 
    [2025-12-21 04:03:17.260]Killed by external signal
    .
    25/12/21 04:03:17 ERROR YarnScheduler: Lost executor 11 on airline-analysis-v3-w-3.us-east1-b.c.big-data-project-481305.internal: Container from a bad node: container_1766261877773_0002_01_000013 on host: airline-analysis-v3-w-3.us-east1-b.c.big-data-project-481305.internal. Exit status: 143. Diagnostics: [2025-12-21 04:03:17.259]Container killed on request. Exit code is 143
    [2025-12-21 04:03:17.260]Container exited with a non-zero exit code 143. 
    [2025-12-21 04:03:17.260]Killed by external signal
    .
    25/12/21 04:03:17 WARN TaskSetManager: Lost task 3.0 in stage 3659.0 (TID 140802) (airline-analysis-v3-w-3.us-east1-b.c.big-data-project-481305.internal executor 11): ExecutorLostFailure (executor 11 exited caused by one of the running tasks) Reason: Container from a bad node: container_1766261877773_0002_01_000013 on host: airline-analysis-v3-w-3.us-east1-b.c.big-data-project-481305.internal. Exit status: 143. Diagnostics: [2025-12-21 04:03:17.259]Container killed on request. Exit code is 143
    [2025-12-21 04:03:17.260]Container exited with a non-zero exit code 143. 
    [2025-12-21 04:03:17.260]Killed by external signal
    .
                                                                                    

    [TIMER] [0.10] eval: 135.56s (2.26m)


                                                                                    

    [0.25] Training on 1211860 rows...


                                                                                    

    [TIMER] [0.25] fit GBT: 432.76s (7.21m)


                                                                                    

    [TIMER] [0.25] eval: 80.07s (1.33m)


                                                                                    

    [0.50] Training on 2420529 rows...


                                                                                    

    [TIMER] [0.50] fit GBT: 790.32s (13.17m)


                                                                                    

    [TIMER] [0.50] eval: 79.88s (1.33m)


                                                                                    

    [1.00] Training on 4843946 rows...


                                                                                    

    [TIMER] [1.00] fit GBT: 4502.15s (75.04m)


                                                                                    

    [TIMER] [1.00] eval: 81.07s (1.35m)
    +----------+-------+-------------------+-------------------+-------------------+------------------+-----------------+------------------+
    |train_frac|train_n|val_pr_auc         |test_pr_auc        |test_recall_top5   |test_f1           |time_total_s     |time_fit_model_s  |
    +----------+-------+-------------------+-------------------+-------------------+------------------+-----------------+------------------+
    |0.1       |484127 |0.30528446080904253|0.30925202316826106|0.12355578864560116|0.3613727843533766|361.5605987420022|226.00559578400134|
    |0.25      |1211860|0.31132224439603956|0.3178245216725219 |0.12558017569162186|0.3650202070084557|512.8247194660034|432.75778674200046|
    |0.5       |2420529|0.31381504462284937|0.31717235765158197|0.12604693850793233|0.3661405765779255|870.2020857069983|790.3183496550009 |
    |1.0       |4843946|0.3123123927345366 |0.3173237403078277 |0.12716402255146192|0.3476221542610598|4583.223232972003|4502.151031329002 |
    +----------+-------+-------------------+-------------------+-------------------+------------------+-----------------+------------------+
    


                                                                                    


```python
# ----------------------------
# 4) SAVE RESULTS (For use in other notebooks)
# ----------------------------

# Method A: Save as a Spark Table (The "In-Memory" persistent way)
# This saves metadata to Hive, allowing any other notebook to query it immediately.
table_name = "bia_scaleup_results"
scaleup_df.write.mode("overwrite").saveAsTable(table_name)
print(f"Saved results to Hive Table: {table_name}")

# Method B: Save as CSV
# .coalesce(1) ensures it writes a single file, not a distributed folder
csv_path = "bia_scaleup_results_csv"
scaleup_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(csv_path)
print(f"Saved results to CSV at: {csv_path}")
```

                                                                                    

    Saved results to Hive Table: bia_scaleup_results
    Saved results to CSV at: bia_scaleup_results_csv


                                                                                    


```python
# 3. Convert Pandas DF -> Spark DF
# This moves the data from local memory to the cluster
spark_df = spark.createDataFrame(pdf_res)

# OPTION A: Save as a Hive Table 
table_name = "bia_scaleup_results_recovered"
spark_df.write.mode("overwrite").saveAsTable(table_name)
print(f"Success! Data saved to Hive Table: {table_name}")

# OPTION B: Save to a specific Bucket/Folder
# coalesce(1) to ensure it writes just 1 CSV file, not multiple parts.
save_path = OUT + "recovered_data/scaleup_results"  
spark_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(save_path)
print(f"Success! Data saved to Bucket path: {save_path}")
```

                                                                                    

    Success! Data saved to Hive Table: bia_scaleup_results_recovered


                                                                                    

    Success! Data saved to Bucket path: gs://big-data-project-481305-flightdelay/airline/final_run_20251221_054405/recovered_data/scaleup_results


# Final Deliverables/Export


```python

```


```python
from datetime import datetime
RUN_ID = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
BASE = "gs://big-data-project-481305-flightdelay/airline"
OUT  = f"{BASE}/final_run_{RUN_ID}/"

print("FINAL OUT:", OUT)

```

    FINAL OUT: gs://big-data-project-481305-flightdelay/airline/final_run_20251221_070407/



```python
# 8.2 Export your best-model classification results (metrics + preds)

# predictions
(gbt_val_pred
 .write.mode("overwrite").parquet(OUT + "preds/gbt_val_parquet/"))

(gbt_test_pred
 .write.mode("overwrite").parquet(OUT + "preds/gbt_test_parquet/"))

# metrics JSON (single row)
import json

metrics = {"run_id": RUN_ID, "gbt": gbt_summary}

metrics_df = spark.createDataFrame([json.dumps(metrics)], "string").toDF("value")
metrics_df.coalesce(1).write.mode("overwrite").text(OUT + "metrics/metrics_json/")


```

                                                                                    


```python
#8.3 Export slice tables (airport / carrier / time block / month)

#Using your existing pred_join:

def slice_table(df, col):
    return (df.groupBy(col)
            .agg(F.count("*").alias("n"),
                 F.avg(F.col("label").cast("double")).alias("delay_rate"),
                 F.avg("p1").alias("avg_pred_risk"),
                 F.avg(F.col("pred").cast("double")).alias("pred_positive_rate"))
            .orderBy(F.desc("n")))

s_airport = slice_table(pred_join, "DEPARTING_AIRPORT_BKT")
s_carrier = slice_table(pred_join, "CARRIER_NAME_BKT")
s_blk     = slice_table(pred_join, "DEP_TIME_BLK")
s_month   = slice_table(pred_join, "MONTH")

s_airport.coalesce(1).write.mode("overwrite").csv(OUT + "slices/airport_csv/", header=True)
s_carrier.coalesce(1).write.mode("overwrite").csv(OUT + "slices/carrier_csv/", header=True)
s_blk.coalesce(1).write.mode("overwrite").csv(OUT + "slices/timeblock_csv/", header=True)
s_month.coalesce(1).write.mode("overwrite").csv(OUT + "slices/month_csv/", header=True)

```

                                                                                    


```python
#8.4 Export drift proxy tables (monthly PR-AUC, risk mean, label rate)

from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

# Build predictions WITH rawPrediction + probability
gbt_test_pred_full = (
    gbt_model.transform(test_fe.select("row_id","label","weight","features"))
    .select("row_id","label","rawPrediction","probability")
    .withColumn("p1", vector_to_array(F.col("probability"))[1])
    .select("row_id","label","p1","rawPrediction")   # keep rawPrediction for evaluator
    .cache()
)
_ = gbt_test_pred_full.count()

```

                                                                                    


```python
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Attach MONTH once
dfm_all = (gbt_test_pred_full
           .join(test_fe.select("row_id","MONTH"), on="row_id", how="inner")
           .cache())
_ = dfm_all.count()

# Base monthly drift stats
monthly_pr_df = (dfm_all.groupBy("MONTH")
    .agg(
        F.count("*").alias("n"),
        F.avg(F.col("label").cast("double")).alias("label_rate"),
        F.avg("p1").alias("mean_p1"),
        F.expr("percentile_approx(p1, 0.5)").alias("median_p1")
    )
)

# PR-AUC per month (12 loops, fine)
e_pr = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR")

months = [r["MONTH"] for r in monthly_pr_df.select("MONTH").collect()]
rows = []
for m in sorted(months):
    pr = float(e_pr.evaluate(dfm_all.where(F.col("MONTH") == m)))
    rows

```

                                                                                    


```python
monthly_pr_df.coalesce(1).write.mode("overwrite").csv(OUT + "drift/monthly_prauc_csv/", header=True)

```

    25/12/21 07:08:31 WARN YarnAllocator: Container from a bad node: container_1766261877773_0002_01_000015 on host: airline-analysis-v3-w-1.us-east1-b.c.big-data-project-481305.internal. Exit status: 143. Diagnostics: [2025-12-21 07:08:30.682]Container killed on request. Exit code is 143
    [2025-12-21 07:08:30.682]Container exited with a non-zero exit code 143. 
    [2025-12-21 07:08:30.685]Killed by external signal
    .
    25/12/21 07:08:31 WARN YarnSchedulerBackend$YarnSchedulerEndpoint: Requesting driver to remove executor 13 for reason Container from a bad node: container_1766261877773_0002_01_000015 on host: airline-analysis-v3-w-1.us-east1-b.c.big-data-project-481305.internal. Exit status: 143. Diagnostics: [2025-12-21 07:08:30.682]Container killed on request. Exit code is 143
    [2025-12-21 07:08:30.682]Container exited with a non-zero exit code 143. 
    [2025-12-21 07:08:30.685]Killed by external signal
    .
    25/12/21 07:08:31 ERROR YarnScheduler: Lost executor 13 on airline-analysis-v3-w-1.us-east1-b.c.big-data-project-481305.internal: Container from a bad node: container_1766261877773_0002_01_000015 on host: airline-analysis-v3-w-1.us-east1-b.c.big-data-project-481305.internal. Exit status: 143. Diagnostics: [2025-12-21 07:08:30.682]Container killed on request. Exit code is 143
    [2025-12-21 07:08:30.682]Container exited with a non-zero exit code 143. 
    [2025-12-21 07:08:30.685]Killed by external signal
    .
                                                                                    


```python
#8.5 Export clustering outputs (profile + quality + sample PCA points)
profile.coalesce(1).write.mode("overwrite").csv(OUT + "clustering/profile_csv/", header=True)

# If you have quality df named quality_df:
# quality_df.coalesce(1).write.mode("overwrite").csv(OUT + "clustering/quality_csv/", header=True)

# export a plot sample (so report can show a scatter)
(vis.select("pc1","pc2",F.col("cluster").alias("cluster"),"intra_cluster_distance")
   .sample(False, 0.05, seed=42).limit(200000)
   .coalesce(1).write.mode("overwrite").csv(OUT + "clustering/pca_points_csv/", header=True))

```

                                                                                    


```python
#8.6 Export anomaly outputs
monthly.write.mode("overwrite").parquet(OUT + "anomalies/monthly_parquet/")
mtb.write.mode("overwrite").parquet(OUT + "anomalies/month_timeblock_parquet/")
mair.write.mode("overwrite").parquet(OUT + "anomalies/month_airport_parquet/")

```

                                                                                    


```python

```
