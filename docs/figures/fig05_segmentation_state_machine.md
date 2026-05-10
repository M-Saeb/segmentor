# Figure 5 — Segmentation Stage State Machine Diagram

**Referenced in**: Section III-F  
**Type**: State machine diagram

## Recommended IEEE Format

**Diagram type**: UML statechart / hierarchical FSM (TikZ `automata`)

Draw rounded rectangles for simple states (Idle, All\_Done) and a large outer rounded rectangle for the composite state `Stage_Active` containing its three inner states (Waiting, Previewing, Confirmed/Skipped). Add a second composite state `Full_Processing` with its own inner chain. Label all transition arcs with the triggering event (user click) and guard/action in a smaller font below. Use a filled circle for the initial pseudo-state and a circle-in-circle for the final pseudo-state.

**LaTeX approach**: TikZ `automata` library — `\node[state]`, `\node[state, accepting]`, `\node[draw, rounded corners, fit=...]` for composite states. Transition labels via `\path[->, bend left/right] ... node[sloped, font=\scriptsize]`. This figure is wide; prefer the two-column `figure*` environment.

---

```mermaid
stateDiagram-v2
    [*] --> Idle

    Idle --> Stage_Active : startup\n(load first chunk, save snapshot)

    state Stage_Active {
        [*] --> Waiting
        Waiting --> Previewing : click "Run Stage"
        Previewing --> Waiting : restore_snapshot()\n→ run RANSAC\n→ update display
        Waiting --> Confirmed : click "Confirm Stage"\nenabled = True
        Waiting --> Skipped : click "Skip Stage"\nenabled = False
    }

    Confirmed --> Stage_Active : advance to next stage\n(save new snapshot)
    Skipped  --> Stage_Active : advance to next stage\n(no snapshot update)

    Stage_Active --> All_Done : no more stages

    All_Done --> Full_Processing : click\n"Finalize & Run"

    state Full_Processing {
        [*] --> Build_Process_Pipeline
        Build_Process_Pipeline --> Iterate_Chunks
        Iterate_Chunks --> Apply_Stages : next spatial tile
        Apply_Stages --> Write_Chunk : append to 2_pipeline.las
        Write_Chunk --> Iterate_Chunks
        Iterate_Chunks --> Write_YAML : all chunks done
        Write_YAML --> [*] : 2_pipeline.yml written
    }

    Full_Processing --> [*] : GUI closes
```
