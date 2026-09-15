import logging
import os

from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils import milvus_util
from knowledge.utils import mongo_import_util

logger = logging.getLogger(__name__)


def delete_document(file_id: str) -> dict:
    record = mongo_import_util.get_import_record(file_id)
    if not record:
        return {"success": False, "message": f"文件记录不存在 (file_id={file_id})"}

    file_title = record.get("file_title", "")
    minio_path = record.get("minio_object_path", "")
    filename = record.get("filename", "")

    milvus_ok = _delete_milvus(file_title)
    mongo_ok = mongo_import_util.delete_import_record(file_id)
    minio_ok = _delete_minio(minio_path)

    success = milvus_ok and mongo_ok and minio_ok
    return {
        "success": success,
        "message": "删除成功" if success else "部分组件删除失败，请检查日志",
        "details": {
            "milvus_chunks": milvus_ok,
            "mongo_record": mongo_ok,
            "minio_object": minio_ok,
        },
        "filename": filename,
    }


def _delete_milvus(file_title: str) -> bool:
    if not file_title:
        return False
    try:
        milvus_client = StorageClients.get_milvus_client()
        collection_name = os.getenv("MILVUS_COLLECTION_NAME", "shopkeeper_brain_knowledge")
        return milvus_util.delete_by_file_title(milvus_client, collection_name, file_title)
    except Exception as e:
        logger.error(f"[delete] Milvus 删除异常: {e}")
        return False


def _delete_minio(object_path: str) -> bool:
    if not object_path:
        return True
    try:
        minio_client = StorageClients.get_minio_client()
        bucket = mongo_import_util.get_bucket_name()
        minio_client.remove_object(bucket, object_path)
        logger.info(f"[delete] MinIO 已删除: {bucket}/{object_path}")
        return True
    except Exception as e:
        logger.error(f"[delete] MinIO 删除异常: {e}")
        return False