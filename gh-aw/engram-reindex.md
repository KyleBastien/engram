# Engram Remote Reindex Workflow

## Triggers

- **Push to main**: Run on every push to the `main` branch
- **Daily schedule**: Run at 2:00 AM UTC every day (`cron: '0 2 * * *'`)

## Prerequisites

- Engram CLI (`engram`) is installed and available on `PATH`
- An embedding provider is configured (e.g., Ollama running locally, or an API key for a remote provider)
- The engram store has been initialized with `engram init --remote <url>`
- Git credentials are available for push access to the store remote

## Workflow Steps

### 1. Clone source repositories

Clone all source repositories defined in `engram.config.yaml` into the workspace.

```bash
engram sync --mode pull --path ./engram-store
```

This pulls the latest store state from the remote, including the config and any previously indexed data.

### 2. Run full reindex

Re-chunk and re-embed all source files to ensure the index is fully up to date.

```bash
engram reindex --full --path ./engram-store
```

Capture the output to determine how many chunks were created, skipped, or deleted.

### 3. Run analysis

Run onboarding analysis to update architectural knowledge and cross-reference data.

```bash
engram onboard --depth standard --path ./engram-store
```

This refreshes the onboarding artifacts (conventions, architecture notes, dependency maps) based on the current source state.

### 4. Flag stale knowledge

Check knowledge items for staleness by comparing their `indexed_at` timestamps against the current date. Items older than 30 days should be flagged for review.

```bash
engram status --path ./engram-store
```

Review the status output for any warnings about stale knowledge entries.

### 5. Commit and push

Stage all changes in the engram store and push to the remote.

```bash
engram sync --mode push --path ./engram-store
```

This commits any new index data, knowledge updates, and analysis artifacts, then pushes to the store remote.

### 6. Open summary PR if >50 chunks updated

If the reindex created or deleted more than 50 chunks in total, open a summary pull request to document the changes.

The PR should include:
- **Title**: `chore: engram reindex summary — <date>`
- **Body**:
  - Number of files processed
  - Number of chunks created, skipped, and deleted
  - Any stale knowledge items flagged
  - Timestamp of the reindex run

Only open a PR if `chunks_created + chunks_deleted > 50`. Otherwise, the sync push from step 5 is sufficient.

## Error Handling

- If `engram sync --mode pull` fails, abort the workflow (store is unreachable)
- If `engram reindex` fails, still attempt to push any partial results and report the error
- If `engram sync --mode push` fails, retry once after 30 seconds; if still failing, report the error

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `ENGRAM_STORE_PATH` | Path to the engram store directory | No (defaults to `./engram-store`) |
| `ENGRAM_EMBEDDING_PROVIDER` | Embedding provider to use | Yes |
| `ENGRAM_EMBEDDING_MODEL` | Model name for the embedding provider | Yes |
| `GIT_AUTHOR_NAME` | Git author for store commits | Yes |
| `GIT_AUTHOR_EMAIL` | Git email for store commits | Yes |
