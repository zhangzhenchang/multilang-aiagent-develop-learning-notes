"""delete.py - 删除日记集合中的数据"""
from milvus_utils import DIARY_COLLECTION_NAME, create_milvus_client, ensure_diary_collection, load_collection


def main() -> None:
    client = create_milvus_client()
    ensure_diary_collection(client)
    load_collection(DIARY_COLLECTION_NAME)

    print("Deleting diary entry...")
    delete_id = "diary_005"
    result = client.delete(collection_name=DIARY_COLLECTION_NAME, filter=f'id == "{delete_id}"')
    print(f"✓ Deleted {result['delete_count']} record(s)")
    print(f"  ID: {delete_id}\n")

    print("Batch deleting diary entries...")
    delete_ids = ["diary_002", "diary_003"]
    ids_str = ", ".join(f'"{item}"' for item in delete_ids)
    batch_result = client.delete(collection_name=DIARY_COLLECTION_NAME, filter=f"id in [{ids_str}]")
    print(f"✓ Batch deleted {batch_result['delete_count']} record(s)")
    print(f"  IDs: {', '.join(delete_ids)}\n")

    print("Deleting by condition...")
    condition_result = client.delete(collection_name=DIARY_COLLECTION_NAME, filter='mood == "sad"')
    print(f"✓ Deleted {condition_result['delete_count']} record(s) with mood=\"sad\"")


if __name__ == "__main__":
    main()
