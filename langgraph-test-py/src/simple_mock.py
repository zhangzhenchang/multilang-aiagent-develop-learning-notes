"""simple_mock.py - 模拟数据：天气查询 & 城市小知识"""
import json

# 模拟天气数据表
_WEATHER_TABLE: dict[str, dict] = {
    "杭州": {"summary": "多云转小雨", "temp_high_c": 22, "temp_low_c": 15, "aqi": "良"},
    "北京": {"summary": "晴", "temp_high_c": 26, "temp_low_c": 12, "aqi": "轻度污染"},
    "上海": {"summary": "阴", "temp_high_c": 20, "temp_low_c": 16, "aqi": "良"},
}

# 模拟城市趣味知识表
_TRIVIA_TABLE: dict[str, str] = {
    "杭州": "西湖文化景观是世界文化遗产之一。",
    "北京": "故宫是世界上现存规模最大的古代宫殿建筑群之一。",
    "上海": "外滩万国建筑博览群是近代城市历史的缩影。",
}


def lookup_weather(city: str) -> str:
    """查某城市当日天气摘要（模拟），返回 JSON 字符串"""
    c = city.strip()
    w = _WEATHER_TABLE.get(c)
    if w is None:
        return json.dumps(
            {"city": c, "summary": "暂无该城市数据，以下为占位", "temp_high_c": 20, "temp_low_c": 12, "aqi": "—"},
            ensure_ascii=False,
        )
    return json.dumps({"city": c, **w}, ensure_ascii=False)


def lookup_city_trivia(city: str) -> str:
    """查与某城市相关的一句小知识（模拟），返回 JSON 字符串"""
    c = city.strip()
    line = _TRIVIA_TABLE.get(c, f"没有为「{c}」准备内置小知识，可换杭州/北京/上海试试。")
    return json.dumps({"city": c, "trivia": line}, ensure_ascii=False)
