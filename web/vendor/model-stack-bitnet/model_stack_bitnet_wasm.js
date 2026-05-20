/* @ts-self-types="./model_stack_bitnet_wasm.d.ts" */

export class AttentionKvCache {
    static __wrap(ptr) {
        const obj = Object.create(AttentionKvCache.prototype);
        obj.__wbg_ptr = ptr;
        AttentionKvCacheFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        AttentionKvCacheFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_attentionkvcache_free(ptr, 0);
    }
    /**
     * @param {Float32Array} q
     * @param {Float32Array} k_new
     * @param {Float32Array} v_new
     * @param {number} q_len
     * @param {boolean} causal
     * @returns {Float32Array}
     */
    append_self_attention(q, k_new, v_new, q_len, causal) {
        const ptr0 = passArrayF32ToWasm0(q, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(k_new, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArrayF32ToWasm0(v_new, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.attentionkvcache_append_self_attention(this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2, q_len, causal);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v4 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v4;
    }
    /**
     * @param {Float32Array} q
     * @param {number} q_len
     * @param {boolean} causal
     * @param {number} past_len
     * @returns {Float32Array}
     */
    attention(q, q_len, causal, past_len) {
        const ptr0 = passArrayF32ToWasm0(q, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.attentionkvcache_attention(this.__wbg_ptr, ptr0, len0, q_len, causal, past_len);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    clear() {
        wasm.attentionkvcache_clear(this.__wbg_ptr);
    }
    /**
     * @returns {AttentionKvCache}
     */
    clone_cache() {
        const ret = wasm.attentionkvcache_clone_cache(this.__wbg_ptr);
        return AttentionKvCache.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    len() {
        const ret = wasm.attentionkvcache_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} n_heads
     * @param {number} head_dim
     */
    constructor(n_heads, head_dim) {
        const ret = wasm.attentionkvcache_new(n_heads, head_dim);
        this.__wbg_ptr = ret;
        AttentionKvCacheFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {Float32Array} k
     * @param {Float32Array} v
     * @param {number} kv_len
     */
    set_cross(k, v, kv_len) {
        const ptr0 = passArrayF32ToWasm0(k, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(v, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.attentionkvcache_set_cross(this.__wbg_ptr, ptr0, len0, ptr1, len1, kv_len);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
}
if (Symbol.dispose) AttentionKvCache.prototype[Symbol.dispose] = AttentionKvCache.prototype.free;

export class BitnetLinearHandle {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        BitnetLinearHandleFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_bitnetlinearhandle_free(ptr, 0);
    }
    /**
     * @param {Uint8Array} packed_weight
     * @param {Float32Array} scale_values
     * @param {Int32Array} segment_offsets
     * @param {Float32Array} bias_values
     * @param {Int32Array} layout_header
     * @param {Float32Array} input_scales
     * @param {number} input_quant_mode
     * @param {number} input_quant_bits
     * @param {number} input_scale_rows
     */
    constructor(packed_weight, scale_values, segment_offsets, bias_values, layout_header, input_scales, input_quant_mode, input_quant_bits, input_scale_rows) {
        const ptr0 = passArray8ToWasm0(packed_weight, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(scale_values, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArray32ToWasm0(segment_offsets, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passArrayF32ToWasm0(bias_values, wasm.__wbindgen_malloc);
        const len3 = WASM_VECTOR_LEN;
        const ptr4 = passArray32ToWasm0(layout_header, wasm.__wbindgen_malloc);
        const len4 = WASM_VECTOR_LEN;
        const ptr5 = passArrayF32ToWasm0(input_scales, wasm.__wbindgen_malloc);
        const len5 = WASM_VECTOR_LEN;
        const ret = wasm.bitnetlinearhandle_new(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4, ptr5, len5, input_quant_mode, input_quant_bits, input_scale_rows);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        BitnetLinearHandleFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {Float32Array} input
     * @param {number} rows
     * @returns {Float32Array}
     */
    run(input, rows) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.bitnetlinearhandle_run(this.__wbg_ptr, ptr0, len0, rows);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
}
if (Symbol.dispose) BitnetLinearHandle.prototype[Symbol.dispose] = BitnetLinearHandle.prototype.free;

export class DecoderLayerHandle {
    static __wrap(ptr) {
        const obj = Object.create(DecoderLayerHandle.prototype);
        obj.__wbg_ptr = ptr;
        DecoderLayerHandleFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        DecoderLayerHandleFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_decoderlayerhandle_free(ptr, 0);
    }
    /**
     * @returns {DecoderLayerHandle}
     */
    clone_cache() {
        const ret = wasm.decoderlayerhandle_clone_cache(this.__wbg_ptr);
        return DecoderLayerHandle.__wrap(ret);
    }
    /**
     * @param {BitnetLinearHandle} self_q
     * @param {BitnetLinearHandle} self_k
     * @param {BitnetLinearHandle} self_v
     * @param {BitnetLinearHandle} self_o
     * @param {BitnetLinearHandle} self_mlp_in
     * @param {BitnetLinearHandle} self_mlp_out
     * @param {BitnetLinearHandle} cross_q
     * @param {BitnetLinearHandle} cross_k
     * @param {BitnetLinearHandle} cross_v
     * @param {BitnetLinearHandle} cross_o
     * @param {BitnetLinearHandle} cross_mlp_in
     * @param {BitnetLinearHandle} cross_mlp_out
     * @param {Float32Array} self_n1_weight
     * @param {Float32Array} self_n1_bias
     * @param {Float32Array} self_n2_weight
     * @param {Float32Array} self_n2_bias
     * @param {Float32Array} cross_n1_weight
     * @param {Float32Array} cross_n1_bias
     * @param {Float32Array} cross_n2_weight
     * @param {Float32Array} cross_n2_bias
     * @param {string} activation
     * @param {number} d_model
     * @param {number} n_heads
     * @param {number} head_dim
     * @param {number} rotary_base
     */
    constructor(self_q, self_k, self_v, self_o, self_mlp_in, self_mlp_out, cross_q, cross_k, cross_v, cross_o, cross_mlp_in, cross_mlp_out, self_n1_weight, self_n1_bias, self_n2_weight, self_n2_bias, cross_n1_weight, cross_n1_bias, cross_n2_weight, cross_n2_bias, activation, d_model, n_heads, head_dim, rotary_base) {
        _assertClass(self_q, BitnetLinearHandle);
        _assertClass(self_k, BitnetLinearHandle);
        _assertClass(self_v, BitnetLinearHandle);
        _assertClass(self_o, BitnetLinearHandle);
        _assertClass(self_mlp_in, BitnetLinearHandle);
        _assertClass(self_mlp_out, BitnetLinearHandle);
        _assertClass(cross_q, BitnetLinearHandle);
        _assertClass(cross_k, BitnetLinearHandle);
        _assertClass(cross_v, BitnetLinearHandle);
        _assertClass(cross_o, BitnetLinearHandle);
        _assertClass(cross_mlp_in, BitnetLinearHandle);
        _assertClass(cross_mlp_out, BitnetLinearHandle);
        const ptr0 = passArrayF32ToWasm0(self_n1_weight, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(self_n1_bias, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArrayF32ToWasm0(self_n2_weight, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passArrayF32ToWasm0(self_n2_bias, wasm.__wbindgen_malloc);
        const len3 = WASM_VECTOR_LEN;
        const ptr4 = passArrayF32ToWasm0(cross_n1_weight, wasm.__wbindgen_malloc);
        const len4 = WASM_VECTOR_LEN;
        const ptr5 = passArrayF32ToWasm0(cross_n1_bias, wasm.__wbindgen_malloc);
        const len5 = WASM_VECTOR_LEN;
        const ptr6 = passArrayF32ToWasm0(cross_n2_weight, wasm.__wbindgen_malloc);
        const len6 = WASM_VECTOR_LEN;
        const ptr7 = passArrayF32ToWasm0(cross_n2_bias, wasm.__wbindgen_malloc);
        const len7 = WASM_VECTOR_LEN;
        const ptr8 = passStringToWasm0(activation, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len8 = WASM_VECTOR_LEN;
        const ret = wasm.decoderlayerhandle_new(self_q.__wbg_ptr, self_k.__wbg_ptr, self_v.__wbg_ptr, self_o.__wbg_ptr, self_mlp_in.__wbg_ptr, self_mlp_out.__wbg_ptr, cross_q.__wbg_ptr, cross_k.__wbg_ptr, cross_v.__wbg_ptr, cross_o.__wbg_ptr, cross_mlp_in.__wbg_ptr, cross_mlp_out.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4, ptr5, len5, ptr6, len6, ptr7, len7, ptr8, len8, d_model, n_heads, head_dim, rotary_base);
        this.__wbg_ptr = ret;
        DecoderLayerHandleFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {Float32Array} input
     * @param {Float32Array} memory
     * @param {number} memory_len
     * @returns {Float32Array}
     */
    next(input, memory, memory_len) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(memory, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.decoderlayerhandle_next(this.__wbg_ptr, ptr0, len0, ptr1, len1, memory_len);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v3;
    }
    /**
     * @returns {number}
     */
    self_len() {
        const ret = wasm.decoderlayerhandle_self_len(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) DecoderLayerHandle.prototype[Symbol.dispose] = DecoderLayerHandle.prototype.free;

export class F5Q4DiTSession {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        F5Q4DiTSessionFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_f5q4ditsession_free(ptr, 0);
    }
    /**
     * @param {string} name
     * @param {Float32Array} values
     */
    add_dense_f32(name, values) {
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(values, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        wasm.f5q4ditsession_add_dense_f32(this.__wbg_ptr, ptr0, len0, ptr1, len1);
    }
    /**
     * @param {string} name
     * @param {Uint8Array} packed_weight
     * @param {Uint16Array} row_scales_f16
     * @param {Float32Array} bias_values
     * @param {number} in_dim
     * @param {number} out_dim
     */
    add_q4_tensor(name, packed_weight, row_scales_f16, bias_values, in_dim, out_dim) {
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray8ToWasm0(packed_weight, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArray16ToWasm0(row_scales_f16, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passArrayF32ToWasm0(bias_values, wasm.__wbindgen_malloc);
        const len3 = WASM_VECTOR_LEN;
        const ret = wasm.f5q4ditsession_add_q4_tensor(this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, in_dim, out_dim);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {number} block
     * @param {Float32Array} input
     * @param {Float32Array} time_embedding
     * @param {number} seq_len
     * @returns {Float32Array}
     */
    debug_dit_block(block, input, time_embedding, seq_len) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(time_embedding, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.f5q4ditsession_debug_dit_block(this.__wbg_ptr, block, ptr0, len0, ptr1, len1, seq_len);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v3;
    }
    /**
     * @param {number} block
     * @param {Float32Array} input
     * @param {Float32Array} time_embedding
     * @param {number} seq_len
     * @returns {string}
     */
    debug_dit_block_profile_json(block, input, time_embedding, seq_len) {
        let deferred4_0;
        let deferred4_1;
        try {
            const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
            const len0 = WASM_VECTOR_LEN;
            const ptr1 = passArrayF32ToWasm0(time_embedding, wasm.__wbindgen_malloc);
            const len1 = WASM_VECTOR_LEN;
            const ret = wasm.f5q4ditsession_debug_dit_block_profile_json(this.__wbg_ptr, block, ptr0, len0, ptr1, len1, seq_len);
            var ptr3 = ret[0];
            var len3 = ret[1];
            if (ret[3]) {
                ptr3 = 0; len3 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred4_0 = ptr3;
            deferred4_1 = len3;
            return getStringFromWasm0(ptr3, len3);
        } finally {
            wasm.__wbindgen_free(deferred4_0, deferred4_1, 1);
        }
    }
    /**
     * @param {Float32Array} input
     * @param {Float32Array} time_embedding
     * @param {number} seq_len
     * @returns {Float32Array}
     */
    debug_final_ada_norm(input, time_embedding, seq_len) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(time_embedding, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.f5q4ditsession_debug_final_ada_norm(this.__wbg_ptr, ptr0, len0, ptr1, len1, seq_len);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v3;
    }
    /**
     * @param {Float32Array} x
     * @param {Float32Array} cond
     * @param {Float32Array} text
     * @param {number} seq_len
     * @param {boolean} drop_audio_cond
     * @returns {Float32Array}
     */
    debug_input_embedding(x, cond, text, seq_len, drop_audio_cond) {
        const ptr0 = passArrayF32ToWasm0(x, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(cond, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArrayF32ToWasm0(text, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.f5q4ditsession_debug_input_embedding(this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2, seq_len, drop_audio_cond);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v4 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v4;
    }
    /**
     * @param {Float32Array} x
     * @param {Float32Array} cond
     * @param {Float32Array} text
     * @param {number} seq_len
     * @param {boolean} drop_audio_cond
     * @returns {string}
     */
    debug_input_embedding_profile_json(x, cond, text, seq_len, drop_audio_cond) {
        let deferred5_0;
        let deferred5_1;
        try {
            const ptr0 = passArrayF32ToWasm0(x, wasm.__wbindgen_malloc);
            const len0 = WASM_VECTOR_LEN;
            const ptr1 = passArrayF32ToWasm0(cond, wasm.__wbindgen_malloc);
            const len1 = WASM_VECTOR_LEN;
            const ptr2 = passArrayF32ToWasm0(text, wasm.__wbindgen_malloc);
            const len2 = WASM_VECTOR_LEN;
            const ret = wasm.f5q4ditsession_debug_input_embedding_profile_json(this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2, seq_len, drop_audio_cond);
            var ptr4 = ret[0];
            var len4 = ret[1];
            if (ret[3]) {
                ptr4 = 0; len4 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred5_0 = ptr4;
            deferred5_1 = len4;
            return getStringFromWasm0(ptr4, len4);
        } finally {
            wasm.__wbindgen_free(deferred5_0, deferred5_1, 1);
        }
    }
    /**
     * @param {Int32Array} text_ids
     * @param {number} seq_len
     * @param {boolean} drop_text
     * @returns {Float32Array}
     */
    debug_text_embedding(text_ids, seq_len, drop_text) {
        const ptr0 = passArray32ToWasm0(text_ids, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.f5q4ditsession_debug_text_embedding(this.__wbg_ptr, ptr0, len0, seq_len, drop_text);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * @param {number} time
     * @returns {Float32Array}
     */
    debug_time_embedding(time) {
        const ret = wasm.f5q4ditsession_debug_time_embedding(this.__wbg_ptr, time);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @param {Float32Array} x
     * @param {Float32Array} cond
     * @param {Int32Array} text_ids
     * @param {number} time
     * @param {boolean} drop_audio_cond
     * @param {boolean} drop_text
     * @returns {Float32Array}
     */
    forward(x, cond, text_ids, time, drop_audio_cond, drop_text) {
        const ptr0 = passArrayF32ToWasm0(x, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(cond, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArray32ToWasm0(text_ids, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.f5q4ditsession_forward(this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2, time, drop_audio_cond, drop_text);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v4 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v4;
    }
    constructor() {
        const ret = wasm.f5q4ditsession_new();
        this.__wbg_ptr = ret;
        F5Q4DiTSessionFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {Float32Array} cond_mel
     * @param {number} cond_seq_len
     * @param {Int32Array} text_ids
     * @param {number} duration
     * @param {number} steps
     * @param {number} cfg_strength
     * @param {number} sway_sampling_coef
     * @param {number} seed
     * @returns {Float32Array}
     */
    sample_mel(cond_mel, cond_seq_len, text_ids, duration, steps, cfg_strength, sway_sampling_coef, seed) {
        const ptr0 = passArrayF32ToWasm0(cond_mel, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(text_ids, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.f5q4ditsession_sample_mel(this.__wbg_ptr, ptr0, len0, cond_seq_len, ptr1, len1, duration, steps, cfg_strength, sway_sampling_coef, seed);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v3;
    }
}
if (Symbol.dispose) F5Q4DiTSession.prototype[Symbol.dispose] = F5Q4DiTSession.prototype.free;

export class Q4LinearHandle {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        Q4LinearHandleFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_q4linearhandle_free(ptr, 0);
    }
    /**
     * @param {Float32Array} input
     * @param {number} rows
     * @returns {Float32Array}
     */
    forward(input, rows) {
        const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.q4linearhandle_forward(this.__wbg_ptr, ptr0, len0, rows);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * @param {Uint8Array} packed_weight
     * @param {Uint16Array} row_scales_f16
     * @param {Float32Array} bias_values
     * @param {number} in_dim
     * @param {number} out_dim
     */
    constructor(packed_weight, row_scales_f16, bias_values, in_dim, out_dim) {
        const ptr0 = passArray8ToWasm0(packed_weight, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray16ToWasm0(row_scales_f16, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArrayF32ToWasm0(bias_values, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.q4linearhandle_new(ptr0, len0, ptr1, len1, ptr2, len2, in_dim, out_dim);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        Q4LinearHandleFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) Q4LinearHandle.prototype[Symbol.dispose] = Q4LinearHandle.prototype.free;

export class TokenSample {
    static __wrap(ptr) {
        const obj = Object.create(TokenSample.prototype);
        obj.__wbg_ptr = ptr;
        TokenSampleFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TokenSampleFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_tokensample_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get probability() {
        const ret = wasm.tokensample_probability(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get rank() {
        const ret = wasm.tokensample_rank(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get token_id() {
        const ret = wasm.tokensample_token_id(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get top_probability() {
        const ret = wasm.tokensample_top_probability(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) TokenSample.prototype[Symbol.dispose] = TokenSample.prototype.free;

/**
 * @param {Float32Array} input
 * @param {string} activation
 * @returns {Float32Array}
 */
export function activate_f32(input, activation) {
    const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(activation, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.activate_f32(ptr0, len0, ptr1, len1);
    var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v3;
}

/**
 * @param {Float32Array} q
 * @param {Float32Array} k
 * @param {Float32Array} v
 * @param {number} q_len
 * @param {number} kv_len
 * @param {number} n_heads
 * @param {number} head_dim
 * @param {boolean} causal
 * @param {number} past_len
 * @returns {Float32Array}
 */
export function attention_f32(q, k, v, q_len, kv_len, n_heads, head_dim, causal, past_len) {
    const ptr0 = passArrayF32ToWasm0(q, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(k, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF32ToWasm0(v, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.attention_f32(ptr0, len0, ptr1, len1, ptr2, len2, q_len, kv_len, n_heads, head_dim, causal, past_len);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v4;
}

/**
 * @param {BitnetLinearHandle} first
 * @param {BitnetLinearHandle} second
 * @param {Float32Array} input
 * @param {number} rows
 * @returns {Float32Array}
 */
export function bitnet_linear2_f32(first, second, input, rows) {
    _assertClass(first, BitnetLinearHandle);
    _assertClass(second, BitnetLinearHandle);
    const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.bitnet_linear2_f32(first.__wbg_ptr, second.__wbg_ptr, ptr0, len0, rows);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v2;
}

/**
 * @param {BitnetLinearHandle} first
 * @param {BitnetLinearHandle} second
 * @param {BitnetLinearHandle} third
 * @param {Float32Array} input
 * @param {number} rows
 * @returns {Float32Array}
 */
export function bitnet_linear3_f32(first, second, third, input, rows) {
    _assertClass(first, BitnetLinearHandle);
    _assertClass(second, BitnetLinearHandle);
    _assertClass(third, BitnetLinearHandle);
    const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.bitnet_linear3_f32(first.__wbg_ptr, second.__wbg_ptr, third.__wbg_ptr, ptr0, len0, rows);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v2;
}

/**
 * @param {Float32Array} input
 * @param {Uint8Array} packed_weight
 * @param {Float32Array} scale_values
 * @param {Int32Array} segment_offsets
 * @param {Float32Array} bias_values
 * @param {Int32Array} layout_header
 * @param {Float32Array} input_scales
 * @param {number} rows
 * @param {number} input_quant_mode
 * @param {number} input_quant_bits
 * @param {number} input_scale_rows
 * @returns {Float32Array}
 */
export function bitnet_linear_f32(input, packed_weight, scale_values, segment_offsets, bias_values, layout_header, input_scales, rows, input_quant_mode, input_quant_bits, input_scale_rows) {
    const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(packed_weight, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF32ToWasm0(scale_values, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArray32ToWasm0(segment_offsets, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ptr4 = passArrayF32ToWasm0(bias_values, wasm.__wbindgen_malloc);
    const len4 = WASM_VECTOR_LEN;
    const ptr5 = passArray32ToWasm0(layout_header, wasm.__wbindgen_malloc);
    const len5 = WASM_VECTOR_LEN;
    const ptr6 = passArrayF32ToWasm0(input_scales, wasm.__wbindgen_malloc);
    const len6 = WASM_VECTOR_LEN;
    const ret = wasm.bitnet_linear_f32(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4, ptr5, len5, ptr6, len6, rows, input_quant_mode, input_quant_bits, input_scale_rows);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v8 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v8;
}

/**
 * @param {BitnetLinearHandle} w_in
 * @param {BitnetLinearHandle} w_out
 * @param {Float32Array} input
 * @param {number} rows
 * @param {string} activation
 * @returns {Float32Array}
 */
export function bitnet_mlp_f32(w_in, w_out, input, rows, activation) {
    _assertClass(w_in, BitnetLinearHandle);
    _assertClass(w_out, BitnetLinearHandle);
    const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(activation, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.bitnet_mlp_f32(w_in.__wbg_ptr, w_out.__wbg_ptr, ptr0, len0, rows, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v3;
}

/**
 * @param {BitnetLinearHandle} lm_head
 * @param {Float32Array} hidden
 * @param {Uint32Array} generated_ids
 * @param {Uint32Array} blocked_ids
 * @param {number} temperature
 * @param {number} top_p
 * @param {number} repetition_penalty
 * @param {number} random_value
 * @returns {TokenSample}
 */
export function bitnet_sample_token_f32(lm_head, hidden, generated_ids, blocked_ids, temperature, top_p, repetition_penalty, random_value) {
    _assertClass(lm_head, BitnetLinearHandle);
    const ptr0 = passArrayF32ToWasm0(hidden, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray32ToWasm0(generated_ids, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArray32ToWasm0(blocked_ids, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.bitnet_sample_token_f32(lm_head.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2, temperature, top_p, repetition_penalty, random_value);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return TokenSample.__wrap(ret[0]);
}

/**
 * @param {Q4LinearHandle} attn_norm
 * @param {Q4LinearHandle} to_q
 * @param {Q4LinearHandle} to_k
 * @param {Q4LinearHandle} to_v
 * @param {Q4LinearHandle} to_out
 * @param {Q4LinearHandle} ff_in
 * @param {Q4LinearHandle} ff_out
 * @param {Float32Array} input
 * @param {Float32Array} time_embedding
 * @param {number} seq_len
 * @param {number} dim
 * @param {number} heads
 * @param {number} head_dim
 * @param {number} eps
 * @returns {Float32Array}
 */
export function f5_dit_block_f32(attn_norm, to_q, to_k, to_v, to_out, ff_in, ff_out, input, time_embedding, seq_len, dim, heads, head_dim, eps) {
    _assertClass(attn_norm, Q4LinearHandle);
    _assertClass(to_q, Q4LinearHandle);
    _assertClass(to_k, Q4LinearHandle);
    _assertClass(to_v, Q4LinearHandle);
    _assertClass(to_out, Q4LinearHandle);
    _assertClass(ff_in, Q4LinearHandle);
    _assertClass(ff_out, Q4LinearHandle);
    const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(time_embedding, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.f5_dit_block_f32(attn_norm.__wbg_ptr, to_q.__wbg_ptr, to_k.__wbg_ptr, to_v.__wbg_ptr, to_out.__wbg_ptr, ff_in.__wbg_ptr, ff_out.__wbg_ptr, ptr0, len0, ptr1, len1, seq_len, dim, heads, head_dim, eps);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v3;
}

/**
 * @param {Float32Array} input
 * @param {number} rows
 * @param {number} cols
 * @param {string} activation
 * @returns {Float32Array}
 */
export function gated_activation_f32(input, rows, cols, activation) {
    const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(activation, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.gated_activation_f32(ptr0, len0, rows, cols, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v3;
}

/**
 * @param {Float32Array} input
 * @param {Float32Array} src
 * @param {Float32Array} gate
 * @param {number} rows
 * @param {number} cols
 * @returns {Float32Array}
 */
export function gated_add_rows_f32(input, src, gate, rows, cols) {
    const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(src, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF32ToWasm0(gate, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.gated_add_rows_f32(ptr0, len0, ptr1, len1, ptr2, len2, rows, cols);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v4;
}

/**
 * @param {Float32Array} input
 * @param {Float32Array} shift
 * @param {Float32Array} scale
 * @param {number} rows
 * @param {number} cols
 * @param {number} eps
 * @returns {Float32Array}
 */
export function layer_norm_affine_f32(input, shift, scale, rows, cols, eps) {
    const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(shift, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF32ToWasm0(scale, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.layer_norm_affine_f32(ptr0, len0, ptr1, len1, ptr2, len2, rows, cols, eps);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v4;
}

/**
 * @param {Float32Array} input
 * @param {Float32Array} weight
 * @param {Float32Array} bias
 * @param {number} rows
 * @param {number} cols
 * @param {number} eps
 * @returns {Float32Array}
 */
export function layer_norm_f32(input, weight, bias, rows, cols, eps) {
    const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(weight, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF32ToWasm0(bias, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.layer_norm_f32(ptr0, len0, ptr1, len1, ptr2, len2, rows, cols, eps);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v4 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v4;
}

/**
 * @param {Float32Array} input
 * @param {Uint8Array} packed_weight
 * @param {Uint16Array} row_scales_f16
 * @param {Float32Array} bias_values
 * @param {number} seq_len
 * @param {number} in_channels
 * @param {number} out_channels
 * @param {number} kernel
 * @param {number} padding
 * @returns {Float32Array}
 */
export function q4_conv1d_f32(input, packed_weight, row_scales_f16, bias_values, seq_len, in_channels, out_channels, kernel, padding) {
    const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(packed_weight, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArray16ToWasm0(row_scales_f16, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF32ToWasm0(bias_values, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.q4_conv1d_f32(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, seq_len, in_channels, out_channels, kernel, padding);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v5;
}

/**
 * @param {Float32Array} input
 * @param {Uint8Array} packed_weight
 * @param {Uint16Array} row_scales_f16
 * @param {Float32Array} bias_values
 * @param {number} seq_len
 * @param {number} channels
 * @param {number} kernel
 * @param {number} padding
 * @returns {Float32Array}
 */
export function q4_depthwise_conv1d_f32(input, packed_weight, row_scales_f16, bias_values, seq_len, channels, kernel, padding) {
    const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(packed_weight, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArray16ToWasm0(row_scales_f16, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF32ToWasm0(bias_values, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.q4_depthwise_conv1d_f32(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, seq_len, channels, kernel, padding);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v5;
}

/**
 * @param {Float32Array} input
 * @param {Uint8Array} packed_weight
 * @param {Uint16Array} row_scales_f16
 * @param {Float32Array} bias_values
 * @param {number} seq_len
 * @param {number} channels
 * @param {number} kernel
 * @param {number} padding
 * @param {number} groups
 * @returns {Float32Array}
 */
export function q4_grouped_conv1d_f32(input, packed_weight, row_scales_f16, bias_values, seq_len, channels, kernel, padding, groups) {
    const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(packed_weight, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArray16ToWasm0(row_scales_f16, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF32ToWasm0(bias_values, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.q4_grouped_conv1d_f32(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, seq_len, channels, kernel, padding, groups);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v5;
}

/**
 * @param {Q4LinearHandle} first
 * @param {Q4LinearHandle} second
 * @param {Q4LinearHandle} third
 * @param {Float32Array} input
 * @param {number} rows
 * @returns {Float32Array}
 */
export function q4_linear3_f32(first, second, third, input, rows) {
    _assertClass(first, Q4LinearHandle);
    _assertClass(second, Q4LinearHandle);
    _assertClass(third, Q4LinearHandle);
    const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.q4_linear3_f32(first.__wbg_ptr, second.__wbg_ptr, third.__wbg_ptr, ptr0, len0, rows);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v2;
}

/**
 * @param {Q4LinearHandle} first
 * @param {Q4LinearHandle} second
 * @param {Float32Array} input
 * @param {number} rows
 * @param {string} activation
 * @returns {Float32Array}
 */
export function q4_mlp_f32(first, second, input, rows, activation) {
    _assertClass(first, Q4LinearHandle);
    _assertClass(second, Q4LinearHandle);
    const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(activation, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.q4_mlp_f32(first.__wbg_ptr, second.__wbg_ptr, ptr0, len0, rows, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v3;
}

/**
 * @param {Float32Array} input
 * @param {Uint8Array} packed_weight
 * @param {Uint16Array} row_scales_f16
 * @param {Float32Array} bias_values
 * @param {number} rows
 * @param {number} in_dim
 * @param {number} out_dim
 * @returns {Float32Array}
 */
export function q4_symmetric_linear_f32(input, packed_weight, row_scales_f16, bias_values, rows, in_dim, out_dim) {
    const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArray8ToWasm0(packed_weight, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArray16ToWasm0(row_scales_f16, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF32ToWasm0(bias_values, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.q4_symmetric_linear_f32(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, rows, in_dim, out_dim);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v5;
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_throw_9c75d47bf9e7731e: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_now_37d9e24e05ce8b2e: function() {
            const ret = Date.now();
            return ret;
        },
        __wbindgen_cast_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./model_stack_bitnet_wasm_bg.js": import0,
    };
}

const AttentionKvCacheFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_attentionkvcache_free(ptr, 1));
const BitnetLinearHandleFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_bitnetlinearhandle_free(ptr, 1));
const DecoderLayerHandleFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_decoderlayerhandle_free(ptr, 1));
const F5Q4DiTSessionFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_f5q4ditsession_free(ptr, 1));
const Q4LinearHandleFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_q4linearhandle_free(ptr, 1));
const TokenSampleFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_tokensample_free(ptr, 1));

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint16ArrayMemory0 = null;
function getUint16ArrayMemory0() {
    if (cachedUint16ArrayMemory0 === null || cachedUint16ArrayMemory0.byteLength === 0) {
        cachedUint16ArrayMemory0 = new Uint16Array(wasm.memory.buffer);
    }
    return cachedUint16ArrayMemory0;
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passArray16ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 2, 2) >>> 0;
    getUint16ArrayMemory0().set(arg, ptr / 2);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedFloat32ArrayMemory0 = null;
    cachedUint16ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('model_stack_bitnet_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
