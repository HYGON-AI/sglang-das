# MiniMax-H3 OpenAPI gap probes

These four dependency-free scripts validate the four reported differences
between `POST /v1/videos` OpenAPI metadata and MiniMax-H3 runtime behavior.
They default to the server address in the MiniMax-H3 launch notes:
`http://127.0.0.1:30000`.

The probes do not enqueue inference. Scripts 01, 02, and 04 send a JSON request
without `task`; MiniMax-H3 rejects it during admission before queue submission.
Script 03 reads an existing job from the in-memory video store, or performs a
schema-only check when `--allow-empty` is supplied.

When testing an uninstalled checkout, put its `python` directory first on
`PYTHONPATH`; otherwise pytest may collect the checkout's tests while importing
an older `sglang` package from `site-packages`:

```bash
PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_video_openapi_contract.py
```

Start the current MiniMax-H3 server and wait until it is healthy:

```bash
curl -fsS http://127.0.0.1:30000/health
curl -fsS http://127.0.0.1:30000/v1/models | python3 -m json.tool
```

## Pre-fix baseline

At commit `c496509fbb`, before the strict H3 schema and HTTP 400 response were
added, the expected states are:

- issue 01: `partial` — the four JSON fields are exposed, but H3-specific
  required fields, task enum, nested `conditions`/`target` schemas, and the
  positive `audio_flow_shift` constraint are not;
- issue 02: `fixed` — `application/json` is declared and accepted;
- issue 03: `fixed` — the stale `sora-2` response default is gone;
- issue 04: `gap` — runtime HTTP 400 string errors remain undocumented.

Run against that historical baseline:

```bash
PROBE_DIR=scripts/ci/utils/diffusion/minimax_h3_openapi_gap
python3 "$PROBE_DIR/verify_01_h3_fields.py" --expect partial
python3 "$PROBE_DIR/verify_02_json_content_type.py" --expect fixed
python3 "$PROBE_DIR/verify_03_response_model.py" --expect fixed --allow-empty
python3 "$PROBE_DIR/verify_04_error_schema.py" --expect gap
```

For a strict schema-versus-runtime check of issue 03, run it after submitting
one of the T2VA requests from the server notes, or pass its job ID explicitly:

```bash
python3 "$PROBE_DIR/verify_03_response_model.py" \
  --expect fixed \
  --video-id "$VIDEO_ID"
```

With the strict H3 OpenAPI contract applied, use `--expect fixed` as regression
mode:

```bash
for script in "$PROBE_DIR"/verify_*.py; do
  python3 "$script" --expect fixed --allow-empty
done
```

All four scripts accept `--allow-empty`, so the same batch command can be used
for each probe. It changes behavior only for issue 03; the other probes ignore
it. The all-`fixed` loop is the expected regression result after applying the
strict H3 request schema and HTTP 400 response contract.

Use `--base-url http://HOST:PORT` for a remote service and `--api-key` (or the
`SGLANG_API_KEY` environment variable) when authentication is enabled.

Exit codes:

- `0`: observed state matches `--expect`;
- `1`: the probe completed, but observed state does not match;
- `2`: inconclusive because the service/schema/job could not be read.
