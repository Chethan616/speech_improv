# Evaluation Evidence Map

| Evaluation requirement | Evidence in the project | Demonstration or verification |
|---|---|---|
| Problem identification | `problem_and_literature.md` defines the degradation model, objectives, constraints, and research gap. | Explain why noise, reverberation, latency, and speech preservation must be considered together. |
| Literature survey of five papers | Linked five-paper survey in `problem_and_literature.md`; detailed comparison in `differences.md`. | Compare the five algorithmic ideas, their real-time properties, and their limitations for this application. |
| Dataset collection and justification | `data/README.md` documents VoiceBank-DEMAND, DNS Challenge, REVERB, controlled mixtures, licence/access considerations, split policy, and manifest procedure. | Show the dataset decision table and one generated JSON manifest for each collected split. |
| System design and workflow diagram | `system_design.md` contains the Mermaid end-to-end system diagram and module responsibilities. | Walk from audio input through buffering, STFT, enhancement, reconstruction, metrics, and downstream ASR. |
| Approximately half-scale implementation milestone | The package implements audio I/O, causal streaming, STFT/ISTFT, noise suppression, dereverberation experiment, controlled degradation, dataset manifests, metrics, and CLI commands. | Run the unit tests and the end-to-end command; explain which larger model/application layers remain extension points. |
| Initial results | `initial_results.md` records measured SNR, SI-SDR, RTF, and nominal latency for the reproducible fixture. | Re-run the command and inspect the generated input/output WAV files and JSON report. |
| Demonstration | `workflow.md` gives the real-data command sequence and failure-case checklist. | Play degraded and enhanced audio, show metrics, report latency, and show an optional ASR comparison when a transcript/ASR model is available. |

## Evidence standard

The included demo is a deterministic voiced-like signal and is suitable for verifying software behavior. It is not a human-speech benchmark. Human-speech claims must be based on a documented dataset split, aligned clean/degraded/enhanced files, and measurements recorded from the actual run.

