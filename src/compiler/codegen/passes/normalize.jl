# IR Normalization Pass
#
# Lowers Julia Core intrinsics and builtins to cuTile Intrinsics in the
# StructuredIRCode. Run immediately after structurization, before all other
# passes. The actual rewrite rules live in rewrite_patterns.jl (NORMALIZE_RULES).

"""
    normalize_ir!(sci::StructuredIRCode)

Replace Julia Core intrinsics with cuTile Intrinsics equivalents using
declarative rewrite rules, then verify no Core intrinsics remain.
"""
function normalize_ir!(sci::StructuredIRCode)
    normalize_pass!(sci)

    # Verify all Core intrinsics were handled
    for block in eachblock(sci)
        for inst in instructions(block)
            call = resolve_call(stmt(inst))
            call === nothing && continue
            func, _ = call
            if func isa Core.IntrinsicFunction
                throw(IRError("Core.Intrinsics.$(nameof(func)) not handled by " *
                              "normalize_ir! — add a rewrite rule in rewrite_patterns.jl or an overlay"))
            end
        end
    end
end
