# SFL Glossary

## Purpose

This glossary supplies a shared, versioned vocabulary for the Systemic Functional Linguistics (SFL) meaning-model build. Each entry gives a plain definition, a read-aloud form, an empirical or computational role, and an implementation reference where one exists.

The glossary supports layered reading. A reader may use the plain definition alone, follow the read-aloud line while reading source, or continue to the data contract and reproducibility record.

## How to Use This Glossary

Code comments and documentation may link to an entry by heading. Each implementation module retains local definitions for its immediate data and model boundary. This glossary maintains the shared project meaning of those definitions.

## Meaning State

**Plain definition:** A meaning state is the structured numerical representation of the SFL meanings recorded for one point in an empirical discourse trajectory.

**Read aloud:** “A meaning state records the configuration of meaning at one point in a discourse trajectory.”

**Current interface:** The initial empirical interface is a 3 by 3 matrix with nine numerical positions.

**Related code:** `olmo/sfl/meaning_state.py`.

## 3 by 3 Meaning-State Interface

**Plain definition:** The 3 by 3 meaning-state interface organises nine numerical positions across the project’s metafunctional and register dimensions.

**Read aloud:** “The meaning-state interface has three rows, three columns, and nine defined positions.”

**Current interface:** Row-major flattening presents the matrix as a vector of length nine for numerical learning.

**Related code:** `olmo/sfl/meaning_state.py`.

## 9D Projection

**Plain definition:** The 9D projection is the current numerical view of a structured meaning state as nine real-valued coordinates.

**Notation:** `z[t] = pi(M[t])`.

**Read aloud:** “z at time t is the nine-dimensional projection of the meaning state M at time t.”

**Data contract:** `z[t]` has shape `(9,)` for one clause-level meaning state and shape `(batch, 9)` for a batch.

## Projection Map pi

**Plain definition:** `pi` is the projection map that presents a structured meaning state through the current nine-dimensional numerical interface.

**Read aloud:** “pi projects a structured meaning state into the current nine-dimensional numerical interface.”

**Signature:** `pi: MeaningState -> R^9`.

## Empirical Projection Coverage

**Plain definition:** Empirical projection coverage records which positions in the 3 by 3 interface receive values from the annotation layers available for a particular dataset release.

**Current dataset coverage:** The initial UAM-derived projection records values in three observed metafunctional positions. The remaining six positions retain zero values in the initial dataset and provide defined positions for later empirically grounded elaboration.

**Read aloud:** “The initial empirical projection records three observed dimensions and retains six defined dimensions for later elaboration.”

## Clause-Level Meaning State

**Plain definition:** A clause-level meaning state is the meaning-state record associated with one annotated clause in a source document.

**Read aloud:** “A clause-level meaning state records the annotated meanings of one clause in its document.”

**Empirical provenance:** UAM CorpusTool annotation layers for transitivity, mood, and theme contribute to the initial dataset interface.

## Successive-Clause Transition

**Plain definition:** A successive-clause transition records the empirically annotated movement from one clause-level meaning state to the succeeding clause-level meaning state in the same source document.

**Read aloud:** “A successive-clause transition records the movement from one clause-level meaning state to the succeeding clause-level meaning state within the same document.”

**Transition record:** `current_meaning_state`, `observed_state_change`, and `next_meaning_state`.

**Current numerical interface:** `(z[t], delta_z[t], z[t + 1])`.

## Transition Observation

**Plain definition:** A transition observation is one recorded successive-clause transition supplied to the learning objective.

**Read aloud:** “A transition observation contains the current meaning state, the observed state change, and the next meaning state.”

**Data contract:** Each numerical item currently has shape `(9,)`.

## Document-Continuity Rule

**Plain definition:** The document-continuity rule constructs transition observations from successive clause-level meaning states belonging to the same source document, ordered by that document’s clause order.

**Read aloud:** “A transition stays within one document and follows that document’s clause order.”

**Dataset role:** Document identifiers and clause identifiers preserve the ordering and provenance required to construct empirical transitions.

## Initial Meaning State

**Plain definition:** The initial meaning state is the first clause-level meaning state recorded for a source document.

**Read aloud:** “The initial meaning state is the first recorded clause-level meaning state in a document.”

**Dataset role:** A document begins with an initial meaning state. Each succeeding clause provides an observed transition from the preceding clause-level meaning state.

## Observed State Change

**Plain definition:** An observed state change is the numerical difference between successive clause-level meaning states in one document.

**Notation:** `delta_z[t] = z[t + 1] - z[t]`.

**Read aloud:** “The observed state change is the next meaning-state projection minus the current meaning-state projection.”

**Data contract:** The initial numerical interface represents an observed state change as a vector with shape `(9,)`.

## Delta-Transition Model

**Plain definition:** A delta-transition model receives a current meaning-state projection and estimates the observed state change to the succeeding clause-level meaning state.

**Notation:** `delta_z_hat[t] = f_theta(z[t])`.

**Read aloud:** “The model estimates the meaning-state change from the current clause to the succeeding clause.”

**Current scope:** The first empirical port-equivalence experiment uses the current state as its model input. Later experiments may extend the input with prior meaning states and empirically recorded contextual conditions.

## Predicted Next Meaning State

**Plain definition:** The predicted next meaning state is the current meaning-state projection plus the model’s estimated state change.

**Notation:** `z_hat[t + 1] = z[t] + delta_z_hat[t]`.

**Read aloud:** “The predicted next meaning state is the current meaning state plus the estimated change.”

## Trajectory Loss

**Plain definition:** Trajectory loss measures agreement between the model’s estimated state change and the observed state change, together with agreement between the predicted next meaning state and the observed next meaning state.

**Current objective:**

`MSE(delta_z_hat[t], delta_z[t]) + 0.5 * (1 - cosine_similarity(z_hat[t + 1], z[t + 1]))`.

**Read aloud:** “Trajectory loss measures agreement in both the meaning-state change and the succeeding meaning state.”

## Empirical UAM Provenance

**Plain definition:** Empirical UAM provenance records the source annotation process and dataset-builder revision from which a trajectory dataset was derived.

**Current source path:** `uam_corpus_ingest.py -> build_uam_dataset.py -> uam_meaning_trajectories.npz`.

**Dataset arrays:** `matrices`, `deltas`, `doc_ids`, and `clause_ids`.

**Metadata record:** `uam_lexical_boundary_index.json` retains document identifiers, clause identifiers, source spans, matrices, and annotation feature lists.

## Random Initialisation

**Plain definition:** Random initialisation creates a new model parameter state from a recorded random seed before empirical trajectory training begins.

**Read aloud:** “Random initialisation creates a new recorded starting state for model learning.”

**Reproducibility role:** A run record retains the seed, model configuration, optimiser configuration, and code revision.

## Checkpoint Lineage

**Plain definition:** Checkpoint lineage records the relationship between a trained model artifact, its dataset artifact, its configuration, its source revision, and its measured training result.

**Read aloud:** “Checkpoint lineage records how a trained model artifact came into being.”

**Baseline lineage:** `uam_corpus_ingest.py -> build_uam_dataset.py -> traincore.py -> sfl_model_3x3.pt`.

## Reproducibility Record

**Plain definition:** A reproducibility record is the versioned set of information required to reconstruct a training result.

**Required fields:**

- repository commit identifier
- dataset artifact path and content hash
- dataset-builder revision
- model configuration
- random seed
- parameter count
- optimiser and scheduler configuration
- epoch count and batch size
- metrics log
- checkpoint path and content hash

**Read aloud:** “A reproducibility record identifies the code, data, settings, measurements, and model artifact for one training result.”

## Delicacy

**Plain definition:** Delicacy is the capacity for a meaning-state position to gain more internally differentiated, empirically grounded structure as the project develops.

**Read aloud:** “Delicacy is the capacity for a meaning dimension to gain more detailed empirical structure.”

**Architecture role:** The initial nine-dimensional interface provides stable positions for later elaboration through additional annotation coverage and structured internal representations.

## Realisation Interface

**Plain definition:** A realisation interface connects a generated meaning-state trajectory with a linguistic expression process.

**Read aloud:** “A realisation interface connects a meaning trajectory with linguistic expression.”

**Project role:** Realisation is a subsequent research component with its own empirical and computational contract.
