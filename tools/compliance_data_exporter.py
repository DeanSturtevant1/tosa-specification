# Copyright (c) 2024-2026, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import os

from tosa import deduce_extensions
from tosa import TOSAOperator
from tosa import TOSAOperatorArgument


validation_term_mapping_profile = {
    "PRO-INT": "Profile::pro_int",
    "PRO-FP": "Profile::pro_fp",
    "EXT-INT64": "Extension::int64",
    "EXT-INT16": "Extension::int16",
    "EXT-INT4": "Extension::int4",
    "EXT-BF16": "Extension::bf16",
    "EXT-FP8E4M3FN": "Extension::fp8e4m3fn",
    "EXT-FP8E5M2": "Extension::fp8e5m2",
    "EXT-FFT": "Extension::fft",
    "EXT-ARGMIN": "Extension::argmin",
    "EXT-AVGPOOL2D-V2": "Extension::avgpool2d_v2",
    "EXT-MATMUL-T": "Extension::matmul_t",
    "EXT-MAXPOOL2D-V2": "Extension::maxpool2d_v2",
    "EXT-SHAPE": "Extension::shape",
    "EXT-VARIABLE": "Extension::variable",
    "EXT-MX-COMMON": "Extension::mx_common",
    "EXT-MX-FP4E2M1FN": "Extension::mx_fp4e2m1fn",
    "EXT-MX-FP6E2M3FN": "Extension::mx_fp6e2m3fn",
    "EXT-MX-FP6E3M2FN": "Extension::mx_fp6e3m2fn",
    "EXT-MX-FP8E4M3FN": "Extension::mx_fp8e4m3fn",
    "EXT-MX-FP8E5M2": "Extension::mx_fp8e5m2",
    "EXT-MX-INT8": "Extension::mx_int8",
    "EXT-CONTROLFLOW": "Extension::controlflow",
}

validation_term_mapping_type = {
    "bool_t": "boolT",
    "i4_t": "i4T",
    "i8_t": "i8T",
    "i16_t": "i16T",
    "i32_t": "i32T",
    "i48_t": "i48T",
    "i64_t": "i64T",
    "bf16_t": "bf16T",
    "fp16_t": "fp16T",
    "fp32_t": "fp32T",
    "fp8e4m3fn_t": "fp8e4m3fnT",
    "fp8e5m2_t": "fp8e5m2T",
    "fp8ue8m0_t": "fp8ue8m0T",
    "fp6e3m2fn_t": "fp6e3m2fnT",
    "fp6e2m3fn_t": "fp6e2m3fnT",
    "fp4e2m1fn_t": "fp4e2m1fnT",
    "mxint8_t": "mxint8T",
    "bs32_fp8ue8m0_fp8e4m3fn_t": "bs32_fp8ue8m0_fp8e4m3fnT",
    "bs32_fp8ue8m0_fp8e5m2_t": "bs32_fp8ue8m0_fp8e5m2T",
    "bs32_fp8ue8m0_fp6e3m2fn_t": "bs32_fp8ue8m0_fp6e3m2fnT",
    "bs32_fp8ue8m0_fp6e2m3fn_t": "bs32_fp8ue8m0_fp6e2m3fnT",
    "bs32_fp8ue8m0_fp4e2m1fn_t": "bs32_fp8ue8m0_fp4e2m1fnT",
    "bs32_fp8ue8m0_mxint8_t": "bs32_fp8ue8m0_mxint8T",
}


def convert_to_export_format_op(op_name: str):
    return "tosa." + op_name.lower()


def convert_to_export_format_profile(profile: str):
    converted_profile = validation_term_mapping_profile.get(profile)
    if not converted_profile:
        raise RuntimeError(f"Invalid profile name {profile}")
    return converted_profile


def convert_to_export_format_type(ty: str):
    converted_ty = validation_term_mapping_type.get(ty)
    if not converted_ty:
        raise RuntimeError(f"Invalid element type name {ty}")
    return converted_ty


def is_matched_print_mode(profile: str, print_mode: str) -> bool:
    if print_mode == "Extension":
        return True if "EXT" in profile else False
    if print_mode == "Profile":
        return True if "EXT" not in profile else False
    raise RuntimeError(f"Invalid printing mode {print_mode}")


# Retrieve the compliance information for the profile-based validation.
def get_profile_compliance_info(
    operator: TOSAOperator, args: TOSAOperatorArgument, print_mode: str
) -> set:
    prof_info = {}

    for tysup in operator.typesupports:
        version_added = tysup.version_added
        base_prof_set = set()  # Should contain all extensions except DEDUCE-EXT
        should_deduce = False

        for profs in tysup.profiles:
            profs = profs.split(" and ")
            for prof in profs:
                if prof == "DEDUCE-EXT":
                    should_deduce = True
                    continue
                if is_matched_print_mode(prof, print_mode):
                    base_prof_set.add(prof)

        for tsmap in tysup.generated_tuples:
            prof_set = set(
                base_prof_set
            )  # Each tsmap can have its own deduced extensions
            if should_deduce:
                deduced_extensions = deduce_extensions(tsmap)
                if print_mode == "Extension":
                    prof_set.update(deduced_extensions)
                # Don't include type entries that also rely on a deduced
                # extension to help avoid duplication. This is because deduced
                # extensions rely on the input/output data types of an operation
                # and presence of such extensions already implies availability
                # of a related profile.
                elif print_mode == "Profile" and len(deduced_extensions) > 0:
                    continue

            if len(prof_set) == 0:
                continue

            prof_list = list(prof_set)
            prof_list.sort()
            prof_str = " ".join(prof_list)

            if prof_str not in prof_info.keys():
                prof_info[prof_str] = []

            # If there are no tensor arguments in the type support map,
            # then skip adding a versioned type support list
            tensor_args = list(
                filter(lambda arg: arg.tensor_element_type in tsmap.keys(), args)
            )
            if len(tensor_args) == 0:
                continue

            entry = (tsmap, version_added)
            if entry not in prof_info[prof_str]:
                prof_info[prof_str].append(entry)

    return prof_info


# Retrieve the profile/extension dependant argurments.
def get_required_arguments_info(operator) -> list:
    # Symbolic types are formal parameters such as in_t, out_t, in_out_t, acc_t, etc.
    symbolic_types = operator.types
    args = []

    for arg in operator.arguments:
        num = len(arg.categories)
        assert num == 1, f"Argument should only have 1 category, but found {num}"

        # First, gather the input and output arguments that have symbolic
        # (non-concrete) types.

        cat = arg.categories[0].name
        if cat == "input" or cat == "output":
            # Concrete types such as mul_t and i8_t, are not necessary for the
            # type indexing.
            if arg.tensor_element_type in symbolic_types:
                args.append(arg)

        # Then, handle uncommon cases where the argument type depends on the
        # profile or extension.

        # `initial_value` of VARIABLE is Attribute type.
        if operator.name == "VARIABLE" and cat == "attribute":
            if arg.tensor_element_type in symbolic_types:
                args.append(arg)

        # `acc_type` of CONV-like ops is Attribute type.
        if arg.name == "acc_type":
            args.append(arg)

    return args


class _PoolBuilder:
    def __init__(self):
        self.items = []
        self.slice_map = {}

    def add_slice(self, seq: tuple) -> tuple[int, int]:
        if not seq:
            return (0, 0)
        if seq in self.slice_map:
            return self.slice_map[seq]
        offset = len(self.items)
        self.items.extend(seq)
        res = (offset, len(seq))
        self.slice_map[seq] = res
        return res


def get_condition(print_mode: str, num_profiles: int) -> str:
    if num_profiles > 1 and print_mode == "Extension":
        return "CheckCondition::allOf"
    return "CheckCondition::anyOf"


def build_argument_compliances(
    args: TOSAOperatorArgument, compliances: list
) -> tuple[tuple[tuple[str, ...], str], ...]:
    vti_list = []
    for compl, version_added in compliances:
        types = []
        for arg in args:
            sym_ty = arg.tensor_element_type

            # Transform enumerated type to symbolic type.
            if sym_ty == "-":
                if arg.type != "acc_type_t":
                    raise RuntimeError(
                        f"Type {arg.type} is not considered in the validation"
                    )
                sym_ty = "acc_t"

            if sym_ty in compl:
                concrete_ty = compl[sym_ty]
                types.append(convert_to_export_format_type(concrete_ty))

        version_major, version_minor = version_added.split(".")
        version_string = f"SpecificationVersion::V_{version_major}_{version_minor}"
        vti_list.append((tuple(types), version_string))

    return tuple(vti_list)


def build_operator_entries(
    operator: TOSAOperator, print_mode: str
) -> tuple[str, tuple] | None:
    args = get_required_arguments_info(operator)
    depot = get_profile_compliance_info(operator, args, print_mode)
    if len(depot) == 0:
        return None

    infos = []
    for profiles, argument_compliances in depot.items():
        post_profiles = profiles.split(" ")
        modes = tuple(convert_to_export_format_profile(p) for p in post_profiles)
        vtis = build_argument_compliances(args, argument_compliances)
        cond = get_condition(print_mode, len(post_profiles))
        infos.append((modes, vtis, cond))

    return (convert_to_export_format_op(operator.name), tuple(infos))


def collect_compliance_entries(spec, print_mode: str) -> list[tuple[str, tuple]]:
    entries = []
    for group in spec.operatorgroups:
        for op in group.operators:
            entry = build_operator_entries(op, print_mode)
            if entry is not None:
                entries.append(entry)
    return sorted(entries, key=lambda x: x[0])


def print_profiles_extensions(spec, outdir):
    prof_entries = collect_compliance_entries(spec, "Profile")
    ext_entries = collect_compliance_entries(spec, "Extension")

    type_pool = _PoolBuilder()
    vti_pool = _PoolBuilder()
    prof_mode_pool = _PoolBuilder()
    ext_mode_pool = _PoolBuilder()

    def build_info_table(entries, mode_pool, mode_array_name):
        info_rows = []
        op_rows = []
        for op_name, infos in entries:
            info_start = len(info_rows)
            for modes, vtis, cond in infos:
                m_off, m_len = mode_pool.add_slice(modes)
                vti_objs = []
                for types, version in vtis:
                    t_off, t_len = type_pool.add_slice(types)
                    vti_objs.append((t_off, t_len, version))
                v_off, v_len = vti_pool.add_slice(tuple(vti_objs))
                info_rows.append((m_off, m_len, v_off, v_len, cond, mode_array_name))
            op_rows.append((op_name, info_start, len(infos)))
        return info_rows, op_rows

    prof_info_rows, prof_op_rows = build_info_table(
        prof_entries, prof_mode_pool, "kProfileModes"
    )
    ext_info_rows, ext_op_rows = build_info_table(
        ext_entries, ext_mode_pool, "kExtensionModes"
    )

    lines = [
        "// The profile-based compliance content below is auto-generated by the script",
        "// `tools/genspec.py` in https://github.com/arm/tosa-specification",
        "",
        "static constexpr TypeInfo kComplianceTypeInfos[] = {",
    ]
    for i in range(0, len(type_pool.items), 8):
        chunk = ", ".join(type_pool.items[i : i + 8])
        lines.append(f"    {chunk},")
    lines.extend(["};", ""])

    lines.append("static constexpr VersionedTypeInfo kVersionedTypeInfos[] = {")
    for t_off, t_len, version in vti_pool.items:
        lines.append(
            f"    {{llvm::ArrayRef(&kComplianceTypeInfos[{t_off}], {t_len}),"
            f" {version}}},"
        )
    lines.extend(["};", ""])

    lines.append("static constexpr Profile kProfileModes[] = {")
    for i in range(0, len(prof_mode_pool.items), 4):
        chunk = ", ".join(prof_mode_pool.items[i : i + 4])
        lines.append(f"    {chunk},")
    lines.extend(["};", ""])

    lines.append("static constexpr Extension kExtensionModes[] = {")
    for i in range(0, len(ext_mode_pool.items), 4):
        chunk = ", ".join(ext_mode_pool.items[i : i + 4])
        lines.append(f"    {chunk},")
    lines.extend(["};", ""])

    def emit_info_array(type_param, array_name, rows):
        lines.append(
            f"static constexpr OpComplianceInfo<{type_param}> {array_name}[] = {{"
        )
        for m_off, m_len, v_off, v_len, cond, mode_arr in rows:
            mode_ref = f"llvm::ArrayRef(&{mode_arr}[{m_off}], {m_len})"
            vti_ref = (
                f"llvm::ArrayRef(&kVersionedTypeInfos[{v_off}], {v_len})"
                if v_len > 0
                else "{}"
            )
            if cond == "CheckCondition::anyOf":
                lines.append(f"    {{{mode_ref}, {vti_ref}}},")
            else:
                lines.append(f"    {{{mode_ref}, {vti_ref}, {cond}}},")
        lines.extend(["};", ""])

    emit_info_array("Profile", "kProfileComplianceInfos", prof_info_rows)
    emit_info_array("Extension", "kExtensionComplianceInfos", ext_info_rows)

    def emit_op_table(type_param, table_name, info_array_name, op_rows):
        lines.append(
            f"static constexpr OpComplianceEntry<{type_param}> {table_name}[] = {{"
        )
        for op_name, i_off, i_len in op_rows:
            lines.append(
                f'    {{"{op_name}",'
                f" llvm::ArrayRef(&{info_array_name}[{i_off}], {i_len})}},"
            )
        lines.extend(["};", ""])

    emit_op_table(
        "Profile",
        "kProfileComplianceTable",
        "kProfileComplianceInfos",
        prof_op_rows,
    )
    emit_op_table(
        "Extension",
        "kExtensionComplianceTable",
        "kExtensionComplianceInfos",
        ext_op_rows,
    )
    lines.append("// End of auto-generated metadata")
    lines.append("")

    with open(os.path.join(outdir, "compliance.meta"), "w") as f:
        f.write("\n".join(lines))
