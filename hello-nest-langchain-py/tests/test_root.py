"""
测试根端点 GET /。

等价于 NestJS e2e test:
    it('/ (GET)', () => request(app).get('/').expect(200).expect('Hello World!'))
"""


def test_get_hello(client):
    """GET / 应返回 200 和 'Hello World!'"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Hello World!"
