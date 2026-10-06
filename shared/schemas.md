# KAIZEN Shared Schemas

This file is the **contract** between all four tracks.
Do not change any field name, type, or format without agreement from all four people in the group chat first. Record every agreed change in the Change Log at the bottom.

---

## Conventions

- **IDs** are strings and never renamed once assigned.
  - Concepts: `c1`, `c2`, `c3`, ...
  - Learners: `l1`, `l2`, ...
- **Scores** are floats in the range `0.0` to `1.0`.
- **Timestamps** are ISO 8601 strings in UTC, e.g. `"2026-10-06T10:30:00Z"`.
- **Field names** are `snake_case`.

---

## 1. Concept Node

One topic in the subject. Owned by Track A.

| Field         | Type    | Notes                                              |
|---------------|---------|----------------------------------------------------|
| `id`          | string  | Stable, e.g. `"c4"`                                |
| `name`        | string  | e.g. `"Loops"`                                     |
| `description` | string  | One or two sentences                               |
| `difficulty`  | integer | 1 (easiest) to 5 (hardest)                         |
| `domain_tag`  | string  | e.g. `"python_fundamentals"`                       |

```json
{
  "id": "c4",
  "name": "Loops",
  "description": "Repeating a block of code using for and while.",
  "difficulty": 2,
  "domain_tag": "python_fundamentals"
}
```

---

## 2. Edge

A "must learn X before Y" link. Owned by Track A.

| Field           | Type    | Notes                                                        |
|-----------------|---------|--------------------------------------------------------------|
| `from_id`       | string  | The prerequisite concept (learn this first)                  |
| `to_id`         | string  | The dependent concept (learn this after)                     |
| `relation_type` | string  | Always `"prerequisite"` for now                              |
| `weight`        | number  | 1 to 10, how strongly `to_id` depends on `from_id`           |

```json
{
  "from_id": "c3",
  "to_id": "c4",
  "relation_type": "prerequisite",
  "weight": 8
}
```

Rules:
- The graph must be a DAG (no cycles).
- Direction is always prerequisite -> dependent.

---

## 3. Mastery Signal

A single score for one learner on one concept, produced after a quiz. Owned by Track B.
**This is the only thing Track C's algorithm is allowed to react to.**

| Field          | Type    | Notes                                          |
|----------------|---------|------------------------------------------------|
| `learner_id`   | string  |                                                |
| `concept_id`   | string  | Must exist in the graph                        |
| `score`        | float   | 0.0 to 1.0                                     |
| `timestamp`    | string  | ISO 8601 UTC                                   |
| `num_questions`| integer | How many questions the score is based on       |

```json
{
  "learner_id": "l1",
  "concept_id": "c4",
  "score": 0.55,
  "timestamp": "2026-10-06T10:30:00Z",
  "num_questions": 5
}
```

---

## 4. Learner State

One learner's progress. Updated whenever a new mastery signal arrives.

| Field                | Type   | Notes                                                       |
|----------------------|--------|-------------------------------------------------------------|
| `learner_id`         | string |                                                             |
| `per_concept_mastery`| object | `{concept_id: score}`, latest score per concept             |
| `history`            | array  | Ordered list of events (see below), oldest first            |

Each `history` entry:

| Field        | Type   | Notes                                                                 |
|--------------|--------|-----------------------------------------------------------------------|
| `timestamp`  | string | ISO 8601 UTC                                                          |
| `concept_id` | string |                                                                       |
| `event`      | string | `"quiz_completed"` or `"concept_recommended"`                         |
| `score`      | float or null | Present for `quiz_completed`, otherwise `null`                 |
| `reason`     | string or null| Present for `concept_recommended`: why it was chosen (explainability log) |

```json
{
  "learner_id": "l1",
  "per_concept_mastery": { "c1": 0.9, "c2": 0.8, "c3": 0.75, "c4": 0.55 },
  "history": [
    {
      "timestamp": "2026-10-06T10:30:00Z",
      "concept_id": "c4",
      "event": "quiz_completed",
      "score": 0.55,
      "reason": null
    },
    {
      "timestamp": "2026-10-06T10:30:01Z",
      "concept_id": "c3",
      "event": "concept_recommended",
      "score": null,
      "reason": "c4 scored 0.55 (< 0.6); revisiting prerequisite c3 before moving on."
    }
  ]
}
```

---

## 5. Shared Interface (Track C, used by both strategies)

Both the static and adaptive strategies implement the same function:

```
next_concept(learner_state, graph) -> { concept_id, reason }
```

- `graph` = list of Concept Nodes + list of Edges
- `learner_state` = Learner State object above
- Returns the id of the concept to learn next, plus a human-readable `reason`.

---

## Change Log

| Date | Change | Agreed by |
|------|--------|-----------|
|      | Initial version | All four |
