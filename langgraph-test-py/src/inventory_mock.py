"""inventory_mock.py - 模拟数据：按 SKU 查库存"""

# 模拟库存数据
_ROWS = [
    {"sku": "SKU-001", "name": "无线鼠标", "stock": 42},
    {"sku": "SKU-002", "name": "机械键盘", "stock": 7},
    {"sku": "SKU-003", "name": "USB-C 线缆", "stock": 120},
]


def get_product_by_sku(sku: str) -> str:
    """根据 SKU 编号查找商品名和库存数量，返回 JSON 字符串"""
    import json

    key = str(sku).strip().upper()
    row = next((r for r in _ROWS if r["sku"].upper() == key), None)
    if row is None:
        return json.dumps({"found": False, "sku": sku.strip()}, ensure_ascii=False)
    return json.dumps({"found": True, **row}, ensure_ascii=False)
