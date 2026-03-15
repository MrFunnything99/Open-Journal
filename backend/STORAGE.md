# Data storage and instance separation

## How storage works

- **Single SQLite DB** at `VECTOR_DB_PATH` (default: `data/open_journal.db`). All clients that talk to the same backend process share this file.
- **No built-in user/account**: There is no login or user ID. Data is keyed only by:
  - **session_id**: A UUID created per browser/session (for chat and for tagging episodic/gist rows). It is **not** used to filter reads—all memory is merged when we run recommendations or show the Memory tab.
  - **instance_id** (optional): When set via the `X-Instance-ID` header, memory and consumed content are scoped to that ID so multiple devices or “instances” can keep separate data on the same backend.

## What’s shared vs per-session

| Data | Where stored | Scope |
|------|----------------|------|
| Chat messages (current conversation) | In-memory `sessions` / `library_interview_sessions` | Per **session_id** (lost on server restart) |
| Gist facts (semantic memory) | `memory_facts` + `vec_gist` | Global unless `instance_id` is set |
| Episodic summaries | `memory_episodic` + `vec_episodic` | Global unless `instance_id` is set |
| Consumed (books, articles, etc.) | `consumed_meta` + `vec_consumed` | Global unless `instance_id` is set |
| People / Brain | `people`, `person_*` | Global (no instance_id yet) |

So **without** `X-Instance-ID`, every device/browser hitting the same backend sees and writes into the **same** memory and library. With `X-Instance-ID` (a stable value per device, e.g. from localStorage), each instance only sees and writes its own slice.

## Why “instances aren’t completely separated”

1. **One backend, multiple devices**: By default all devices share one DB and one in-memory session store. Only the current chat buffer is per session_id; persistent memory and consumed are global.
2. **Same DB file**: If you run the backend on two machines but point both at the same `VECTOR_DB_PATH` (e.g. synced folder), they share the same DB and can see mixed or conflicting data.
3. **Session ID is not an instance boundary**: `session_id` is stored on episodic/gist rows for reference but is **not** used to filter what you see. So all sessions’ data is combined for recommendations and Memory.

## Optional: instance isolation

Set a stable **instance ID** per device (e.g. a UUID in localStorage) and send it as the **`X-Instance-ID`** header on API requests. The backend then:

- Writes new memory and consumed rows with that `instance_id`
- Reads memory and consumed only for that `instance_id`

Existing data with no instance_id (legacy) is treated as shared; new data with an instance_id is isolated. This lets you test or use multiple “profiles” or devices against the same backend without mixing data.
