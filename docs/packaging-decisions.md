# tracefork Packaging: Best Practices for Optional Framework Integrations

*Synthesis of six adversarially-vetted research dimensions on Python packaging for optional, extra-gated framework integrations. Every load-bearing claim below is tied to a primary source; where the underlying evidence was **DISPUTED** or **NUANCED**, that is called out inline rather than smoothed over.*

---

## 1. TL;DR verdict

- **Keep per-integration PEP 621 extras — they are mandatory, not stylistic.** PEP 735 dependency-groups are *structurally unreachable* via `pip install tracefork[adk]` (they are excluded from published wheel/sdist metadata and have no install interface), so extras are the only bracket-installable mechanism for user-facing integrations ([PEP 735, Final](https://peps.python.org/pep-0735/); [Dependency Groups spec](https://packaging.python.org/en/latest/specifications/dependency-groups/)). Do **not** split the ~9 thin, first-party, observer-only adapters into separate `tracefork-*` distributions — tracefork has none of the conditions (scale, decoupled cadence, distributed ownership) that drove the famous splits ([Airflow AIP-8](https://cwiki.apache.org/confluence/display/AIRFLOW/AIP-8+Split+Providers+into+Separate+Packages+for+Airflow+2.0); [LangChain v0.1](https://www.langchain.com/blog/langchain-v0-1-0)).
- **Add a *curated* `all` extra via a self-reference — but deliberately exclude the mutually-heavy framework stacks.** A single god-`all` over five independently-capped, fast-moving agent frameworks is a fragility and bloat anti-pattern ([PEP 771 bloat guidance](https://peps.python.org/pep-0771/)). Ship `all = ["tracefork[providers,bedrock,mcp,observability]"]` and leave the framework stacks to their own extras. Self-referential extras are well-supported and, under hatchling, flattened into concrete metadata at build time ([Hynek Schlawack](https://hynek.me/articles/python-recursive-optional-dependencies/); [core-metadata thread](https://discuss.python.org/t/core-metadata-for-self-referential-extras/77793)).
- **Drop the *speculative* `<2`/`<3` caps; keep caps only on genuinely churny/pre-stable deps, each with an inline rationale + revisit note.** In Python's flat graph a library upper-cap cannot be overridden downstream and is permanent once on PyPI ([Schreiner](https://iscinumpy.dev/post/bound-version-constraints/); [PyPA install_requires](https://packaging.python.org/en/latest/discussions/install-requires-vs-requirements/)). Replace the `langchain-core<2` cap with a floor + targeted `!=`; keep single-major caps on `openai-agents` (0.x), `autogen`, `crewai`, and `google-adk` **only** because their adapters inject into private/undocumented internals — Schreiner's blessed exception.
- **Upgrade the runtime guard to `raise ImportError(HINT) from exc`.** Keep the lazy `X_available()`/`require_X()` pair and the exact-extra-name hint; the one high-confidence, do-it-now change is preserving the chained cause so an *installed-but-broken* dependency's real error survives ([Real Python](https://realpython.com/python-raise-exception/); [Python exception chaining](https://docs.python.org/3/library/exceptions.html)).
- **Migrate `dev` to `[dependency-groups]` — but treat it as a deliberate, later, breaking change,** not a free win: it removes `uv sync --extra dev` and every call site in `CLAUDE.md`/`README`/`scripts/e2e.sh` must move to `--group dev` in the same commit ([PEP 735](https://peps.python.org/pep-0735/); [pip 25.1 `--group`](https://ichard26.github.io/blog/2025/04/whats-new-in-pip-25.1/)).

---

## 2. Per-dimension best practice

### 2.1 Standards & mechanism (extras vs PEP 735 dependency-groups)

**Best practice: user-facing optional integrations MUST be `[project.optional-dependencies]` (extras); development-only tooling belongs in `[dependency-groups]`.** This is not a style preference — it is a structural constraint. The finalized spec states build backends **"MUST NOT include Dependency Group data in built distributions as package metadata"** and that **"there is no syntax or specification-defined interface for installing or referring to dependency groups"** ([Dependency Groups spec](https://packaging.python.org/en/latest/specifications/dependency-groups/); [PEP 735, Status: Final](https://peps.python.org/pep-0735/)). Extras, by contrast, are published to `METADATA` (`Provides-Extra`/`Requires-Dist`) and installable via `pkg[extra]`. pyOpenSci frames the division cleanly: optional-dependencies are for functionality "a user wants to access," dependency-groups "organize development dependencies and are intentionally separate" ([pyOpenSci](https://www.pyopensci.org/python-package-guide/package-structure-code/declare-dependencies.html)).

*Verdict: **SUPPORTED**, unqualified.* tracefork's langchain/crewai/adk/provider adapters are exactly the user-facing case, so they must stay extras. Do **not** design around [PEP 771 default-extras](https://peps.python.org/pep-0771/) — still **Draft** (round-2 discussion June 2025), targeting an unshipped Metadata-Version 2.5, with only unreviewed proof-of-concept branches. Keep the base install minimal and every integration strictly opt-in.

### 2.2 The `[all]` convenience extra

**Best practice: a self-referential meta-extra is a legitimate, well-precedented pattern — but curate it to an internally-consistent family, don't god-`all` heavyweight independent stacks.** The self-reference (`all = ["tracefork[providers,mcp,...]"]`) is an ordinary PEP 508 requirement naming the project itself, not special grammar; setuptools discussion #3627 confirms "that is perfectly OK," and dask ships exactly this in production: `complete = ["dask[array,dataframe,distributed,diagnostics]", ...]` ([Hynek](https://hynek.me/articles/python-recursive-optional-dependencies/); [dask pyproject.toml](https://raw.githubusercontent.com/dask/dask/main/pyproject.toml); [setuptools #3627](https://github.com/pypa/setuptools/discussions/3627)).

**NUANCED — mechanism correction:** the common framing that "the backend passes metadata through unchanged; the pip-21.2 resolver does the recursion" is **backwards for hatchling**. Charlie Marsh's demonstration shows hatchling **FLATTENS** the self-reference into concrete `Requires-Dist` at *build* time (setuptools preserves it), so a hatchling wheel's METADATA already contains the concrete union and pip never performs the recursion ([core-metadata thread](https://discuss.python.org/t/core-metadata-for-self-referential-extras/77793)). The end result is equivalent, but declare the self-ref knowing the backend resolves it. Paul Moore's caveat in that thread — "extras are badly specified by the standards" — is why the post-edit resolution check is required, not optional.

**NUANCED — the god-`all` warning is directional, not a proven unsolvable build:** the provable downsides of unioning tracefork's five capped, fast-moving frameworks (langchain, crewai, autogen, adk, openai-agents) are (a) **union-fragility** — one future cap collision breaks the whole `all` install though each extra installs fine alone — and (b) **install/vuln bloat**, which PEP 771 explicitly warns to avoid ("carefully consider what is included... to avoid unnecessarily bloating installations") ([PEP 771](https://peps.python.org/pep-0771/)). The claim that these frameworks are "not co-resolvable today" is **unverified** and should not be asserted — modern resolvers (uv) are strong, and running a real solve would violate the offline/$0 invariant. State the fragility mechanism, not a certainty.

*Not-triggered caveats worth documenting:* the [hatch #1610](https://github.com/pypa/hatch/issues/1610) self-ref bug is real but scoped to the `hatch env` *manager* — tracefork uses `uv sync`/`uv run`, never `hatch env`, so it cannot bite; and uv's own historical self-ref bug ([astral-sh/uv #1987](https://github.com/astral-sh/uv/issues/1987)) is long fixed. Still, **verify `uv sync --extra all` resolves after the edit** — recursive extras have documented lock/compile edge cases ([pip-tools #2002](https://github.com/jazzband/pip-tools/issues/2002)).

### 2.3 Extras vs separate packages

**Best practice: keep integrations as extras until an adapter independently needs decoupled release cadence, gains an external co-maintainer, or develops a conflicting hard dependency — never split on integration *count*.** The decisive precondition behind every famous split was **scale + independent cadence + dependency isolation**, not count. Airflow's AIP-8 split providers for release-cadence and distributed-expertise reasons and *explicitly named the cost*: **"in the long-term structure there are many more packages to maintain"** ([AIP-8](https://cwiki.apache.org/confluence/display/AIRFLOW/AIP-8+Split+Providers+into+Separate+Packages+for+Airflow+2.0)). LangChain split at **700+ integrations** to fix install headaches, enable per-package breaking-change versioning, and de-bloat an unstable core with CVE exposure ([LangChain v0.1](https://www.langchain.com/blog/langchain-v0-1-0); [v0.2](https://blog.langchain.com/langchain-v02-leap-to-stability/)).

**NUANCED — the load-bearing reason is scale/cadence, not "no external partners."** Airflow's own docs confirm many providers are **community-maintained**, so external ownership is a *common but not necessary* trigger ([apache-airflow-providers](https://airflow.apache.org/docs/apache-airflow-providers/)). tracefork lacks the scale (~9 vs 58–700) and any cadence-decoupling need regardless. At its profile — solo maintainer, guarded imports already containing blast radius to opt-in users — a split's benefits are near-zero and AIP-8's named cost (N repos, N pipelines, an N×N core-vs-adapter compatibility matrix) lands in full.

*Future-proofing:* register an `importlib.metadata` entry-point group (e.g. `tracefork.adapters`) resolved by `adapters/base.py`, so a *later* extraction of any single adapter is non-breaking.

### 2.4 Naming / normalization

**Best practice: lowercase-hyphen extra names — which tracefork already uses — are PEP 685/PEP 503 conformant; no rename is required.** PEP 685 mandates that extras be compared and *written* normalized (`re.sub(r"[-_.]+", "-", name).lower()`) and that generators error if two extras collide ([PEP 685](https://peps.python.org/pep-0685/); [Name normalization](https://packaging.python.org/en/latest/specifications/name-normalization/)). Hatchling normalizes extra names by default ([Hatch metadata docs](https://hatch.pypa.io/latest/config/metadata/)). tracefork's ten names (`dev, providers, bedrock, mcp, frameworks, openai-agents, crewai, autogen, adk, observability`) are all-lowercase, already-normalized, and collision-free — the MUST-error branch cannot fire.

**Three framing corrections (none change the KEEP verdict):**
- The transparent underscore↔hyphen equivalence is a **post-685-tooling property only**. The [huggingface_hub #3029](https://github.com/huggingface/huggingface_hub/issues/3029) bug shows pip ≤ 22.0.2 *silently skipped* a mismatched `hf_transfer` extra; pip 25.0.1+ tolerates it. This "doubly matters" for a self-referential `all`: both the extra name and the bracket must normalize consistently.
- Hatchling's `allow-ambiguous-features` opt-out is **deprecated-but-still-present**, not removed — soften any "removed" wording. tracefork never sets it, so this is moot.
- "lowercase kebab-case mirroring the distribution name" is a *consequence of normalization plus community practice*, not a convention the Packaging User Guide prescribes ([writing-pyproject-toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) prescribes none).

**The real user-facing gotcha is the shell, not TOML.** Bare TOML keys allow dashes, so `openai-agents = [...]` needs no quotes ([TOML v1.0.0](https://toml.io/en/v1.0.0)) — only a **dot** would force quoting. But zsh (default macOS shell) globs `[...]`, so unquoted `pip install tracefork[dev]` fails with `zsh: no matches found` ([napari #2081](https://github.com/napari/napari/issues/2081); [Berton](https://lucaberton.medium.com/how-to-fix-zsh-no-matches-found-when-installing-python-packages-with-optional-dependencies-40aec463ff09)). **Always quote bracketed install specs in README/CLAUDE.md.**

### 2.5 Version caps

**Best practice: floor everywhere + targeted `!=` for proven-bad releases; reserve blanket upper caps for genuinely unstable deps that *also* carry a documented lift-the-cap cadence.** The consensus (Brett Cannon via Schreiner, PyPA, pip docs) is robust: "set a floor... but otherwise don't cap the maximum version as you can't predict future compatibility," and crucially **"you can't fix an over-constraint"** — an over-cap is unfixable downstream while an under-constraint is a trivial user workaround ([Schreiner](https://iscinumpy.dev/post/bound-version-constraints/)). PyPA's install_requires guide is author-facing and squarely agrees: preemptive pinning is "overly-restrictive, and prevents the user from gaining the benefit of dependency upgrades" ([PyPA](https://packaging.python.org/en/latest/discussions/install-requires-vs-requirements/)). Because PyPI artifacts are immutable, a bad cap in a shipped release is **permanent** ([prefix.dev](https://prefix.dev/blog/the_python_packaging_debate); [ResolutionImpossible](https://pip.pypa.io/en/stable/ux-research-design/resolution-impossible-example/)).

**Schreiner's blessed exceptions apply to tracefork's adapters — this is genuinely NUANCED, and the nuance *sharpens* the recommendation:**
- Caps are acceptable for "an extension for an ecosystem/framework (pytest/Sphinx/Jupyter extension)... capping on the major version" and when "you depend on private internal details of a library." tracefork's defensive-attribute-injection adapters (langchain/openai-agents/crewai/autogen/adk) fit both.
- **But a single-major cap gives ZERO protection against the real observed failure mode.** Schreiner's own words: private-internal breakage "can be broken in a minor or patch release, and often is." Verified in the wild: `langchain-core` shipped a 1.x patch regression breaking `ToolsRenderer`/`RetrieverInput` imports, and `langgraph-prebuilt` shipped breaking changes in patch releases **1.0.2** (required a new `runtime` param — [#6363](https://github.com/langchain-ai/langgraph/issues/6363)), **1.0.5** ([#6477](https://github.com/langchain-ai/langgraph/issues/6477)), and **1.0.9**. A `<2` cap catches *none* of these — they live inside 1.x. **Only a targeted `!=` catches them, and CI-against-latest is the actual guard; the cap is a courtesy hint.**

**Per-dependency calls:**
- **`langchain-core >=1.4,<2` → DROP to `>=1.4`.** LangChain 1.0 GA (Oct 22 2025) commits to "no breaking changes until 2.0" ([release policy](https://docs.langchain.com/oss/python/release-policy)), so `<2` is speculative *and* useless against the real intra-1.x breakage — keep `!=` in the toolkit, monitor.
- **`autogen-* >=0.7,<1` → KEEP.** Genuinely experimental/pre-stable: 0.4 is a ground-up async rewrite, officially "expect changes and bugs," now fracturing into Microsoft's Agent Framework ([migration guide](https://microsoft.github.io/autogen/0.4.8//user-guide/agentchat-user-guide/migration-guide.html)).
- **`google-adk >=2.3,<3` → KEEP, but correct the rationale.** ADK is **not** pre-1.0 — v1.0.0 shipped May 2025 as "stable, production-ready." The cap rests on **fast major cadence + adapter-internal coupling**, not "pre-1.0 instability," and is only honest paired with a CI item that tests the newest major each release.
- **`openai-agents >=0.10,<1` → KEEP** (0.x, pre-stable API, defensive internal injection).
- **`crewai >=1.0,<2` → KEEP defensibly** (adapter hooks LiteLLM's private `client_session` internals) *with* a revisit note.
- **`providers` (openai, google-genai), `bedrock` (boto3), `mcp`, `observability` (structlog, opentelemetry-*) → floors already; leave uncapped.** Stable-wire-format / self-instrumentation deps, not private-internal framework extensions.

*Caveat:* tracefork uses hatchling/PEP 621, so it never *inherits* Poetry-style auto-caps ([poetry-relax](https://pypi.org/project/poetry-relax/); [poetry #2731](https://github.com/python-poetry/poetry/issues/2731)) — these caps are hand-written and deliberate, which is exactly why each needs an inline `# TODO: widen` comment.

### 2.6 Runtime UX

**Best practice: keep the lazy `X_available()` + `require_X()` guard pair naming the exact project extra; upgrade to chain the cause with `from exc`.** The idiom — import lazily, catch `ImportError`, re-raise a *new* `ImportError` naming the remediation — is confirmed against scikit-learn's `check_matplotlib_support`: `raise ImportError(f"... You can install matplotlib with 'pip install matplotlib'") from e` ([sklearn](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_optional_dependencies.py)). **DISPUTED exemplar:** sklearn is *inconsistent* — its sibling `check_pandas_support` raises only `"... requires pandas."` with no remediation. This **strengthens** tracefork's design: naming the exact quoted extra in *every* adapter is strictly better than sklearn's mixed messages.

`raise ... from exc` is the documented default for wrapping/transforming exceptions and preserves the original traceback ([Real Python](https://realpython.com/python-raise-exception/); [Python docs](https://docs.python.org/3/library/exceptions.html)). It is the safer default than `from None` because it does **not** hide an *installed-but-broken* dependency's real error — which, given tracefork injects into undocumented framework internals, is the exact confusing `AttributeError` you want surfaced. This is "best practice," not "strictly required" (`from None` is defensible for the pure missing-package case).

**Optional (not load-bearing):** a pandas-style `require_extra(module, extra)` DRY helper. Worthwhile only if adapters grow; for ~5 adapters the duplication is small. **Do NOT adopt `lazy_loader`;** PEP 562 `__getattr__` is optional polish, not needed here.

---

## 3. Concrete recommendation for tracefork

**Decisions, sequenced.**

### For the next 0.2.1 release (do now — pyproject + docs + one runtime idiom)

**1. Keep per-integration extras.** No structural change; they are the only bracket-installable mechanism.

**2. Relax the speculative caps to floors; keep only the internal-coupling caps, each with an inline rationale + revisit note.** Example shape:

```toml
[project.optional-dependencies]
# stable wire-format / self-instrumentation — floors only
providers     = ["openai>=1.0", "google-genai>=0.3"]
bedrock       = ["boto3>=1.34"]
mcp           = ["mcp>=1.0"]
observability = ["structlog>=24.1", "opentelemetry-api>=1.27", "opentelemetry-sdk>=1.27"]

# LangChain 1.0 GA promises no breaks until 2.0 -> floor only
# (a <2 cap is useless against the real intra-1.x patch regressions)
frameworks = ["langchain-core>=1.4", "langchain-openai>=1.3", "langchain-anthropic>=1.4", "langgraph>=1.2"]

# CAPS: adapter injects into private/undocumented internals (Schreiner's blessed
# exception). Each cap is a hint, NOT a guard — CI-against-latest is the guard.
# TODO(caps): test newest major each release and widen.
openai-agents = ["openai-agents>=0.10,<1"]   # 0.x pre-stable API
autogen       = ["autogen-core>=0.7,<1", "autogen-ext>=0.7,<1"]  # experimental rewrite
crewai        = ["crewai>=1.0,<2"]           # hooks LiteLLM private client_session
adk           = ["google-adk>=2.3,<3"]       # stable but fast-major + internal coupling

# curated convenience extra — self-referential, deliberately EXCLUDES the
# mutually-heavy framework stacks and dev
all = ["tracefork[providers,bedrock,mcp,observability]"]
```

**Self-referencing extras are safe here.** Under hatchling the `all` self-reference is flattened into concrete `Requires-Dist` at build time. **Required post-edit step (not a formality):** run `uv sync --extra all` and `uv run pytest -q` to confirm resolution and that the offline suite still passes.

*Why exclude the framework stacks from `all`:* unioning five independently-capped, fast-moving frameworks makes `all` a single all-or-nothing failure target — one future cap collision `ResolutionImpossible`s the whole install though each extra installs fine alone. The curated `all` mirrors dask's internally-consistent `complete`, not pandas' god-`all`.

**3. Upgrade the runtime guard to chain the cause.** Sketch:

```python
# tracefork/adapters/base.py
def require_extra(module: str, extra: str):
    """Import `module` or raise an actionable, chained ImportError."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # keep `from exc` — preserves an installed-but-broken dep's real error
        raise ImportError(
            f"tracefork's {extra!r} integration needs {module!r}. "
            f"Install it with:  pip install 'tracefork[{extra}]'"
        ) from exc
```

**4. Docs:** quote every bracketed install snippet in `README`/`CLAUDE.md` (`pip install 'tracefork[all]'`), and add a small **extras table** disambiguating the generic names.

**5. Register an `entry-points` group** (`tracefork.adapters`) resolved by `adapters/base.py`, so a future single-adapter extraction is non-breaking.

### Later (deliberate, higher-churn changes)

**6. Migrate `dev` to `[dependency-groups]`** — breaking to the dev command surface; update `CLAUDE.md`/`README`/`scripts/e2e.sh` to `uv sync --group dev` in the same commit.

**7. Add runtime version/feature checks** to the internal-coupling adapters (the real guard the caps only hint at).

### Explicitly decided against

- **No `tracefork-*` package split.** Split trigger = an adapter needing decoupled cadence, a co-maintainer, or a conflicting hard dependency — never count.
- **No god-`all` over the frameworks.**
- **No rename of extras for conformance** — already PEP 685/503 clean.

---

## 4. What NOT to do

- **Do NOT put user-facing integrations in `[dependency-groups]`.** Excluded from published metadata, no bracket-install interface.
- **Do NOT design around PEP 771 default-extras.** Still Draft, no shipped tooling.
- **Do NOT ship a single god-`all` over the five fast-moving frameworks.**
- **Do NOT keep preemptive `<2`/`<3` caps on stable-wire deps.** A library upper-cap can't be overridden downstream and is permanent once on PyPI.
- **Do NOT treat a single-major cap as protection against private-internal breakage.** Those ship in minor/patch releases inside the allowed range.
- **Do NOT let a caught `ImportError` swallow the original.** Re-raise with `from exc`.
- **Do NOT leave unquoted bracket install commands in docs.** zsh globs them.
- **Do NOT rename extras to add underscores or rely on old-pip normalization.**
- **Do NOT split adapters into separate packages "for cleanliness."**
- **Do NOT migrate `dev` off extras as a "quick additive edit."**
- **Do NOT reach for `lazy_loader` or PEP 562 `__getattr__`** — unnecessary for ~5 adapters.

---

*Method: 13-agent research workflow (6 dimensions → adversarial verification → synthesis), all claims web-sourced against primary references. Generated 2026-07-03.*
