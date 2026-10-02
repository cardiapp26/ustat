"""The output document exported as one Word file."""
import base64
import io
import struct
import zlib

import pytest

docx = pytest.importorskip("docx")

def _png() -> str:
    """A valid 1x1 RGB PNG, built rather than typed in."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    raw = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)) \
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00")) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(raw).decode()


PNG = _png()


def _items():
    return [
        {
            "title": "Multinomial Logistic Regression",
            "meta": "models · 2 Oct 2026",
            "note": "Reference: paroxysmal.",
            "blocks": [
                {"type": "paragraph", "text": "N: 300"},
                {"type": "heading", "text": "Likelihood-ratio tests", "level": 4},
                {"type": "table", "header_rows": 1,
                 "rows": [["Predictor", "LR χ²", "df", "p"], ["age", "26.88", "2", "<0.001"]]},
            ],
        },
        {
            "title": "Fine-Gray Competing Risks",
            "blocks": [{"type": "image", "image": PNG}, {"type": "paragraph", "text": "CIF by arm."}],
        },
    ]


def _post(client, items, **kw):
    return client.post("/api/pub_export/output_docx", json={"title": "uSTAT output: trial.csv", "items": items, **kw})


def test_items_arrive_in_order_with_tables_and_figures(client):
    r = _post(client, _items(), filename="trial_output")
    assert r.status_code == 200, r.text
    assert 'filename="trial_output.docx"' in r.headers["content-disposition"]
    doc = docx.Document(io.BytesIO(r.content))
    text = [p.text for p in doc.paragraphs]
    order = [text.index(t) for t in (
        "uSTAT output: trial.csv", "Multinomial Logistic Regression", "Reference: paroxysmal.",
        "N: 300", "Likelihood-ratio tests", "Fine-Gray Competing Risks", "CIF by arm.",
    )]
    assert order == sorted(order)
    assert len(doc.tables) == 1
    table = doc.tables[0]
    assert [c.text for c in table.rows[1].cells] == ["age", "26.88", "2", "<0.001"]
    assert table.rows[0].cells[0].paragraphs[0].runs[0].bold is True
    assert len(doc.inline_shapes) == 1


@pytest.mark.parametrize("image", [
    "https://evil.example/x.png",
    "data:image/svg+xml;base64,PHN2Zz4=",
    "data:image/png;base64,not-base64!!",
    # Valid base64, PNG signature, then garbage: must be a 422, not a 500.
    "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40).decode(),
])
def test_figures_must_be_embedded_png_or_jpeg(client, image):
    r = _post(client, [{"title": "x", "blocks": [{"type": "image", "image": image}]}])
    assert r.status_code == 422


def test_unknown_block_types_are_rejected(client):
    r = _post(client, [{"title": "x", "blocks": [{"type": "html", "text": "<script>"}]}])
    assert r.status_code == 422


def test_empty_document_is_refused(client):
    assert _post(client, []).status_code == 422


def test_filename_is_reduced_to_safe_characters(client):
    r = _post(client, _items(), filename='a"b/../c\r\nX-Evil: 1')
    assert r.status_code == 200
    disposition = r.headers["content-disposition"]
    assert "\r" not in disposition and "/" not in disposition and '"b' not in disposition
