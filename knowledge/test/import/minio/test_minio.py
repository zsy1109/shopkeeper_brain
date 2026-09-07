import os
from minio import Minio
from minio.error import S3Error

def main():
    # 1. 实例化MinIO客户端
    client = Minio(os.getenv("MINIO_ENDPOINT", "localhost:9000"),
                   access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
                   secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
                   secure=False
                   )

    source_file = r"C:\path\to\your\file.png"

    # 3. 桶名
    bucket_name = "python-test-bucket"

    # 4. 对象名字
    destination_file = "my-test-png.png"

    # 5. 判断桶是否存在
    found = client.bucket_exists(bucket_name)
    if not found:
        # 5.1 创建桶
        client.make_bucket(bucket_name)
        print("Created bucket", bucket_name)
    else:
        print("Bucket", bucket_name, "already exists")

    # 6. 上传文件
    client.fput_object(
        bucket_name, destination_file, source_file,
    )
    print("上传成功")


if __name__ == "__main__":
    try:
        main()
    except S3Error as exc:
        print("error occurred.", exc)