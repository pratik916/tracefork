# tracefork-plugin-example

A minimal, standalone example of a third-party [`tracefork`](https://github.com/pratik916/tracefork)
`RequestMatcher` plugin, written against the public extension API documented in
[`docs/plugin-api.md`](../../docs/plugin-api.md).

`NonceStrippingMatcher` (`plugin_example/matcher.py`) canonicalizes a request by
dropping one volatile per-call header (`x-request-nonce`) before hashing — the
same "normalize before matching" pattern tracefork's own built-in
`CanonicalizingMatcher` uses for Gemini/Bedrock — while still satisfying the
`RequestMatcher` round-trip invariant:

```
stored_fingerprint(stored_request(R)) == live_fingerprint(R)
```

This package is **not** installed or imported by tracefork itself; it exists
purely to demonstrate the shape a real plugin package takes and is exercised
by `tests/test_plugin_example.py` in the main repo (added to `sys.path`
directly, no build/install step needed for that).

## Try it for real

```bash
pip install -e .            # from this directory
export TRACEFORK_ALLOW_PLUGINS=example_nonce_stripping
python -c "
from tracefork.matcher import load_matcher_entry_points, get_matcher
load_matcher_entry_points()
print(get_matcher('example_nonce_stripping'))
"
```

Nothing loads without that explicit allowlist — see `docs/plugin-api.md`'s
security model.
