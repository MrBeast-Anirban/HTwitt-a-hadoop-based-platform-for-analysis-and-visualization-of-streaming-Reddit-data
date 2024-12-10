# src/verify_hadoop_data.py

from pyspark.sql import SparkSession
import os
import subprocess

def check_hadoop_status():
    """Check if Hadoop is running and directories exist"""
    try:
        # Check if HDFS is accessible
        result = subprocess.run(['hdfs', 'dfs', '-ls', '/'], 
                              capture_output=True, 
                              text=True)
        
        if result.returncode != 0:
            print("Error: HDFS is not accessible. Starting Hadoop services...")
            subprocess.run(['start-dfs.sh'])
            subprocess.run(['start-yarn.sh'])
            print("Waiting for services to start...")
            import time
            time.sleep(10)
    except Exception as e:
        print(f"Error checking Hadoop status: {str(e)}")
        return False
    return True

def create_hdfs_directories():
    """Create necessary HDFS directories"""
    try:
        commands = [
            'hdfs dfs -mkdir -p /user',
            'hdfs dfs -mkdir -p /user/landslide',
            'hdfs dfs -mkdir -p /user/landslide/raw_data',
            'hdfs dfs -chmod -R 777 /user/landslide'
        ]
        
        for cmd in commands:
            subprocess.run(cmd.split(), check=True)
        print("HDFS directories created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating HDFS directories: {str(e)}")
        return False

def verify_hadoop_data():
    """Verify and display data stored in Hadoop"""
    
    # Check Hadoop status
    if not check_hadoop_status():
        print("Error: Hadoop is not running properly")
        return
    
    # Create directories if they don't exist
    # if not create_hdfs_directories():
    #     print("Error: Could not create HDFS directories")
    #     return
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("LandslideDataVerification") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
        .getOrCreate()
    
    try:
        # Check if data exists
        try:
            df = spark.read.parquet("hdfs:///user/landslide/raw_data")
            
            # Display basic information
            print("\n=== Data Overview ===")
            print(f"Total number of records: {df.count()}")
            print("\nSchema:")
            df.printSchema()
            
            print("\n=== Sample Data ===")
            df.show(5, truncate=False)
            
        except Exception as e:
            print("\nNo data found in HDFS. This is normal if you haven't run the data collection yet.")
            print("To add data, run the main.py script first.")
            print("\nHDFS directory structure:")
            subprocess.run(['hdfs', 'dfs', '-ls', '-R', '/user/landslide'])
            
    except Exception as e:
        print(f"Error verifying data: {str(e)}")
    
    finally:
        spark.stop()

if __name__ == "__main__":
    verify_hadoop_data()