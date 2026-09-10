"""Queue behaviour: enqueue, status translation, and the inline fallback."""

import fakeredis
import pytest
from rq import Queue, SimpleWorker

from doc_agent import config, jobs


def fake_ingest(path: str, chunk_tokens: int, overlap_tokens: int) -> dict:
    """Stand in for ingest_document. RQ imports jobs by path, so this must be
    an importable module-level function rather than a lambda or closure."""
    return {
        "source": "doc.pdf", "pages": 2, "chunks": 5,
        "total_tokens": 900, "average_chunk_tokens": 180,
    }


def failing_ingest() -> dict:
    """A job that raises, to exercise the failure path."""
    raise RuntimeError("the PDF was unreadable")


@pytest.fixture
def fake_redis(monkeypatch):
    """Point the jobs module at an in-process Redis substitute."""
    server = fakeredis.FakeStrictRedis()
    monkeypatch.setattr(jobs, "connection", lambda: server)
    monkeypatch.setattr(config, "REDIS_URL", "redis://fake:6379/0")
    return server


def test_queue_is_disabled_without_a_redis_url(monkeypatch):
    monkeypatch.setattr(config, "REDIS_URL", None)

    assert jobs.queue_enabled() is False


def test_queue_is_enabled_with_a_redis_url(fake_redis):
    assert jobs.queue_enabled() is True


def test_enqueue_returns_a_job_that_is_queued(fake_redis, tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")

    job_id = jobs.enqueue_ingest(pdf)
    status = jobs.job_status(job_id)

    assert status.state == "queued"
    assert status.finished is False
    assert "waiting for a worker" in status.detail


def test_an_unknown_job_id_is_reported_not_crashed(fake_redis):
    status = jobs.job_status("no-such-job")

    assert status.state == "failed"
    assert "expired" in status.detail


def test_a_worker_runs_the_job_and_the_result_comes_back(fake_redis):
    """End to end through RQ, with ingestion stubbed so no model is loaded."""
    queue = Queue(jobs.QUEUE_NAME, connection=fake_redis)
    job = queue.enqueue(fake_ingest, "doc.pdf", 400, 60)

    SimpleWorker([queue], connection=fake_redis).work(burst=True)

    status = jobs.job_status(job.id)
    assert status.state == "finished"
    assert status.finished is True
    assert status.result["chunks"] == 5


def test_a_failing_job_reports_failure_rather_than_raising(fake_redis):
    queue = Queue(jobs.QUEUE_NAME, connection=fake_redis)
    job = queue.enqueue(failing_ingest)

    SimpleWorker([queue], connection=fake_redis).work(burst=True)

    status = jobs.job_status(job.id)
    assert status.state == "failed"
    assert status.finished is True


def test_states_are_plain_strings_not_enum_reprs(fake_redis, tmp_path):
    """RQ returns an enum; a user must never see 'JobStatus.QUEUED'."""
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")

    status = jobs.job_status(jobs.enqueue_ingest(pdf))

    assert status.state == "queued"
    assert "JobStatus" not in status.state
    assert "JobStatus" not in status.detail


def test_finished_state_is_also_a_plain_string(fake_redis):
    queue = Queue(jobs.QUEUE_NAME, connection=fake_redis)
    job = queue.enqueue(fake_ingest, "doc.pdf", 400, 60)
    SimpleWorker([queue], connection=fake_redis).work(burst=True)

    status = jobs.job_status(job.id)

    assert status.state == "finished"
    assert "JobStatus" not in status.state
