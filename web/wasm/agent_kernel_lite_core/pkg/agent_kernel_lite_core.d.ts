/* tslint:disable */
/* eslint-disable */

export class AgentLiteCore {
    free(): void;
    [Symbol.dispose](): void;
    can_continue(): boolean;
    compile_context_packet(task_json: string, evidence_json: string, history_json: string): string;
    export_episode_json(): string;
    finish_model_reply(model_text: string): string;
    finish_turn(assistant_text: string): string;
    install_extension_manifest(manifest_json: string): string;
    list_extension_manifests(): string;
    constructor(session_id: string, mode: string, max_context_items: number);
    parse_model_decision(model_text: string, options_json: string): string;
    plan_lite_turn(user_text: string, history_json: string, options_json: string): string;
    propose_extension_action(extension_id: string, capability_id: string, input_json: string): string;
    propose_last_decision_extension_action(input_json: string): string;
    rank_evidence(query: string, candidate_rows_json: string, limit: number): string;
    record_extension_result(action_id: string, receipt_json: string): string;
    register_extension_manifest(manifest_json: string): string;
    reset(): void;
    runtime_attestation(): string;
    set_extension_enabled(extension_id: string, enabled: boolean): string;
    set_mode(mode: string): void;
    snapshot_json(): string;
    start_turn(user_text: string, context_rows_json: string, language: string, max_new_tokens: number): string;
    start_turn_with_context(task_json: string, retrieval_candidates_json: string, history_json: string, options_json: string): string;
    step_count(): number;
    uninstall_extension(extension_id: string): string;
}

export function image_ternary_linear_f32(input: Float32Array, weight_values: Int8Array, scale_values: Float32Array, bias_values: Float32Array, rows: number, in_dim: number, out_dim: number): Float32Array;

export function image_ternary_packed_2bit_linear_f32(input: Float32Array, packed_weight_values: Uint8Array, scale_values: Float32Array, bias_values: Float32Array, rows: number, in_dim: number, out_dim: number): Float32Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly image_ternary_linear_f32: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number, number, number];
    readonly image_ternary_packed_2bit_linear_f32: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number, number, number];
    readonly __wbg_agentlitecore_free: (a: number, b: number) => void;
    readonly agentlitecore_new: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly agentlitecore_set_mode: (a: number, b: number, c: number) => void;
    readonly agentlitecore_reset: (a: number) => void;
    readonly agentlitecore_step_count: (a: number) => number;
    readonly agentlitecore_can_continue: (a: number) => number;
    readonly agentlitecore_start_turn: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number];
    readonly agentlitecore_finish_turn: (a: number, b: number, c: number) => [number, number];
    readonly agentlitecore_finish_model_reply: (a: number, b: number, c: number) => [number, number];
    readonly agentlitecore_parse_model_decision: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly agentlitecore_plan_lite_turn: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly agentlitecore_rank_evidence: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
    readonly agentlitecore_compile_context_packet: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly agentlitecore_start_turn_with_context: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number];
    readonly agentlitecore_register_extension_manifest: (a: number, b: number, c: number) => [number, number];
    readonly agentlitecore_install_extension_manifest: (a: number, b: number, c: number) => [number, number];
    readonly agentlitecore_uninstall_extension: (a: number, b: number, c: number) => [number, number];
    readonly agentlitecore_set_extension_enabled: (a: number, b: number, c: number, d: number) => [number, number];
    readonly agentlitecore_list_extension_manifests: (a: number) => [number, number];
    readonly agentlitecore_propose_extension_action: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly agentlitecore_propose_last_decision_extension_action: (a: number, b: number, c: number) => [number, number];
    readonly agentlitecore_record_extension_result: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly agentlitecore_export_episode_json: (a: number) => [number, number];
    readonly agentlitecore_runtime_attestation: (a: number) => [number, number];
    readonly agentlitecore_snapshot_json: (a: number) => [number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
