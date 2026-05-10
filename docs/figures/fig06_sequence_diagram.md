# Figure 6 — Sequence Diagram: "Run Stage" to "Finalize"

**Referenced in**: Section III-F.3  
**Type**: UML sequence diagram

## Recommended IEEE Format

**Diagram type**: UML sequence diagram (pgf-umlsd or TikZ manual)

Draw five vertical lifelines: *User*, *PipelineViewer*, *PipelineController*, *LasPipelineWrapper*, *pcd\_tools*. Group the interaction into three clearly labeled `Note` boxes spanning all lifelines: *Preview cycle*, *Confirm and advance*, *Finalize*. Use solid arrows for calls, dashed arrows for returns, and an activation box (thin rectangle) on each lifeline while it is executing. The `loop` frame around the per-chunk iteration in the Finalize section should be a standard UML combined fragment.

**LaTeX approach**: Use the `pgf-umlsd` package (`\begin{sequencediagram}`) which handles lifeline spacing automatically. Alternatively, draw manually in TikZ with `\draw[dashed]` lifelines and `\draw[->]` messages. Use `\begin{figure*}` (two-column span) — five lifelines are too wide for a single column.

---

```mermaid
sequenceDiagram
    actor User
    participant Viewer  as PipelineViewer
    participant Control as PipelineController
    participant Wrapper as LasPipelineWrapper
    participant Tools   as pcd_tools

    Note over User,Tools: ── Preview cycle (repeated per parameter tweak) ──

    User   ->> Viewer  : click "Run Stage"
    Viewer ->> Viewer  : show_loading()
    Viewer ->> Control : run_current_stage_of_setup_pipeline()
    activate Control
    Control ->> Wrapper : restore_snapshot()
    Wrapper -->> Control : (pcd, points_attributes) restored
    Control ->> Tools   : segment_ground() / apply_ceiling_segmentation() / apply_wall_segmentation()
    Tools   -->> Control : (pcd, points_attributes) updated
    Control ->> Wrapper : set_current_chunk(pcd, attrs)
    deactivate Control
    Control -->> Viewer : post_to_main_thread(_on_done)
    Viewer  ->> Viewer  : update_preview_pcd()
    Viewer  ->> Viewer  : hide_loading()

    Note over User,Tools: ── Confirm and advance to next stage ──

    User    ->> Viewer  : click "Confirm Stage"
    Viewer  ->> Control : set_flag_for_current_setup_stage(True)
    Viewer  ->> Control : confirm_current_stage_and_advance()
    Control ->> Wrapper : save_snapshot()
    Control -->> Viewer : next PipelineStage info
    Viewer  ->> Viewer  : reconfigure UI for next stage

    Note over User,Tools: ── Finalize: apply confirmed params to all chunks ──

    User    ->> Viewer  : click "Finalize & Run Full Processing"
    Viewer  ->> Viewer  : show_loading()
    Viewer  ->> Control : build_process_pipeline_from_setup()
    Viewer  ->> Control : run_process_pipeline_on_all_chunks()

    loop for each spatial tile
        Control ->> Wrapper : iter_input_chunks()
        Wrapper -->> Control : (index, pcd, attrs)
        Control ->> Tools   : stage.func(pcd, attrs, **params)
        Tools   -->> Control : (pcd, attrs) classified
        Control ->> Wrapper : write_pcb_chunk_to_file()
    end

    Control -->> Viewer : post_to_main_thread(_on_done)
    Viewer  ->> Viewer  : show_message("Done", output_path)
```
