from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, explode, sequence, datediff, lit, sum as sum_, year, month, dayofmonth

# Step 1: Start SparkSession
spark = SparkSession.builder \
    .appName("Accurate Daily CUR Service Cost Breakdown") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.2,com.amazonaws:aws-java-sdk-bundle:1.11.901") \
    .getOrCreate()

# Step 2: Read CUR Parquet files
input_paths = [
    "s3a://data-input-inbox/1mg-edp/cur-hourly-athena-ttn-1mg-edp-aispl/cur-hourly-athena-ttn-1mg-edp-aispl/year=2025/month=2/",
    "s3a://data-input-inbox/1mg-edp/cur-hourly-athena-ttn-1mg-edp-aispl/cur-hourly-athena-ttn-1mg-edp-aispl/year=2025/month=1/",
    "s3a://data-input-inbox/1mg-edp/cur-hourly-athena-ttn-1mg-edp-aispl/cur-hourly-athena-ttn-1mg-edp-aispl/year=2025/month=3/",
    "s3a://data-input-inbox/1mg-edp/cur-hourly-athena-ttn-1mg-edp-aispl/cur-hourly-athena-ttn-1mg-edp-aispl/year=2024/month=10/",
    "s3a://data-input-inbox/1mg-edp/cur-hourly-athena-ttn-1mg-edp-aispl/cur-hourly-athena-ttn-1mg-edp-aispl/year=2024/month=11/",
    "s3a://data-input-inbox/1mg-edp/cur-hourly-athena-ttn-1mg-edp-aispl/cur-hourly-athena-ttn-1mg-edp-aispl/year=2024/month=12/"
]
df = spark.read.parquet(*input_paths)

# Step 3: Select and prepare data
df = df.select(
    to_date("line_item_usage_start_date").alias("start_date"),
    to_date("line_item_usage_end_date").alias("end_date"),
    col("product_product_name").alias("service_name"),
    col("line_item_blended_cost").cast("double").alias("cost")
).filter(
    col("cost").isNotNull() & (col("cost") > 0)
)

# Step 4: Calculate days spanned and daily cost
df = df.withColumn(
    "days_spanned", datediff(col("end_date"), col("start_date")) + lit(1)
).withColumn(
    "daily_cost", col("cost") / col("days_spanned")
)

# Step 5: Explode each usage period into individual days
df = df.withColumn(
    "usage_date", explode(sequence(col("start_date"), col("end_date")))
)

# Step 6: Add year, month, day columns
df = df.withColumn("year", year("usage_date")) \
       .withColumn("month", month("usage_date")) \
       .withColumn("day", dayofmonth("usage_date"))

# Step 7: Aggregate daily cost by service and date components
result_df = df.groupBy("service_name", "year", "month", "day").agg(
    sum_("daily_cost").alias("total_cost")
).orderBy("year", "month", "day", "service_name")

# Step 8: Show and write result
result_df.show(truncate=False)

# Optional: Write as Parquet (partitioning on year, month, day)
result_df.write.mode("overwrite").partitionBy("year", "month", "day").parquet("s3a://data-input-inbox/cur-daily-service-costs/")

# Optional: Write as flat CSV
result_df.write.mode("overwrite").option("header", True).csv("s3a://data-input-inbox/cur-daily-service-costs-csv/")

print("✅ Output written to S3 with separate year, month, day columns.")

