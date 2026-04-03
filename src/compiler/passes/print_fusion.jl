# Print Fusion Pass
#
# Fuses format_string intrinsic calls into print_tko calls to support
# Julia's string interpolation in kernel print statements.
#
# Julia lowers `print("hello $x")` to `print(string("hello ", x))`.
# The overlay routes `string(xs...)` → `Intrinsics.format_string(xs...)`
# and `print(xs...)` → `Intrinsics.print_tko(xs...)`.
#
# This pass inlines the format_string args into the print_tko call:
#
#   %1 = call format_string("hello ", %x)
#   %2 = call print_tko(%1, "\n")
#
# becomes:
#
#   %2 = call print_tko("hello ", %x, "\n")
#
# Dead format_string calls are removed by subsequent DCE.

"""
    print_fusion_pass!(sci::StructuredIRCode)

Inline `format_string` args into `print_tko` calls to support string interpolation.
"""
function print_fusion_pass!(sci::StructuredIRCode)
    print_fusion_block!(sci.entry)
end

function print_fusion_block!(block::Block)
    # First, recurse into nested control flow
    for inst in instructions(block)
        s = stmt(inst)
        if s isa ForOp
            print_fusion_block!(s.body)
        elseif s isa IfOp
            print_fusion_block!(s.then_region)
            print_fusion_block!(s.else_region)
        elseif s isa WhileOp
            print_fusion_block!(s.before)
            print_fusion_block!(s.after)
        elseif s isa LoopOp
            print_fusion_block!(s.body)
        end
    end

    # Collect format_string definitions: SSA index → operand args
    format_defs = Dict{Int, Vector{Any}}()
    for inst in instructions(block)
        call = resolve_call(block, stmt(inst))
        call === nothing && continue
        func, operands = call
        func === Intrinsics.format_string || continue
        format_defs[inst.ssa_idx] = collect(operands)
    end

    isempty(format_defs) && return

    # Find print_tko calls and splice in format_string args
    for inst in instructions(block)
        call = resolve_call(block, stmt(inst))
        call === nothing && continue
        func, _ = call
        func === Intrinsics.print_tko || continue

        s = stmt(inst)
        # Determine operand offset: :call → args[2:], :invoke → args[3:]
        arg_start = s.head === :invoke ? 3 : 2

        # Scan args in reverse to splice without invalidating indices
        i = length(s.args)
        while i >= arg_start
            arg = s.args[i]
            if arg isa SSAValue && haskey(format_defs, arg.id)
                fmt_args = format_defs[arg.id]
                splice!(s.args, i, fmt_args)
                i = i + length(fmt_args) - 1  # skip the newly-inserted args
            end
            i -= 1
        end
    end
end
