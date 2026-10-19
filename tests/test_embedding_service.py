from types import SimpleNamespace

from app import embedding_service


def test_embedding_stays_on_gemini_embedding_model(monkeypatch):
    calls = []
    fake_response = SimpleNamespace(
        embeddings=[SimpleNamespace(values=[0.1, 0.2, 0.3])]
    )

    def fake_embed_content(**kwargs):
        calls.append(kwargs)
        return fake_response

    monkeypatch.setattr(
        embedding_service.client.models,
        "embed_content",
        fake_embed_content,
    )

    result = embedding_service.generate_embedding("sample text")

    assert result == [0.1, 0.2, 0.3]
    assert calls == [{
        "model": "gemini-embedding-001",
        "contents": "sample text",
    }]
