#!/usr/bin/env python3
# Copyright (c) 2024-2026, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import re

op_list = [
    "argmax",
    "argmin",
    "avg_pool2d",
    "avg_pool2d_v2",
    "conv2d",
    "conv3d",
    "depthwise_conv2d",
    "fft2d",
    "fully_connected",
    "matmul",
    "max_pool2d",
    "max_pool2d_v2",
    "rfft2d",
    "transpose_conv2d",
    "clamp",
    "erf",
    "sigmoid",
    "tanh",
    "add",
    "arithmetic_right_shift",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "intdiv",
    "logical_and",
    "logical_left_shift",
    "logical_right_shift",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
    "mul",
    "pow",
    "sub",
    "table",
    "abs",
    "bitwise_not",
    "ceil",
    "clz",
    "cos",
    "exp",
    "floor",
    "log",
    "logical_not",
    "negate",
    "reciprocal",
    "rsqrt",
    "select",
    "sin",
    "equal",
    "greater",
    "greater_equal",
    "reduce_all",
    "reduce_any",
    "reduce_max",
    "reduce_min",
    "reduce_product",
    "reduce_sum",
    "concat",
    "pad",
    "reshape",
    "reverse",
    "slice",
    "tile",
    "transpose",
    "gather",
    "scatter",
    "resize",
    "cast",
    "rescale",
    "const",
    "identity",
    "custom",
    "cond_if",
    "while_loop",
    "variable",
    "variable_write",
    "variable_read",
    "add_shape",
    "concat_shape",
    "const_shape",
    "dim",
    "div_shape",
    "mul_shape",
    "sub_shape",
    "matmul_t",
    "assert_equal_shape",
    "div_ceil_shape",
    "div_floor_shape",
    "exp2_shape",
    "log2_ceil_shape",
    "log2_floor_shape",
    "max_shape",
    "min_shape",
    "mod_shape",
    "slice_shape",
]

profile_list = [
    "Profile::pro_int",
    "Profile::pro_fp",
    "Extension::int64",
    "Extension::int16",
    "Extension::int4",
    "Extension::bf16",
    "Extension::fp8e4m3fn",
    "Extension::fp8e5m2",
    "Extension::fft",
    "Extension::argmin",
    "Extension::avgpool2d_v2",
    "Extension::matmul_t",
    "Extension::maxpool2d_v2",
    "Extension::variable",
    "Extension::shape",
    "Extension::mx_common",
    "Extension::mx_fp4e2m1fn",
    "Extension::mx_fp6e2m3fn",
    "Extension::mx_fp6e3m2fn",
    "Extension::mx_fp8e4m3fn",
    "Extension::mx_fp8e5m2",
    "Extension::mx_int8",
    "Extension::controlflow",
]

type_list = [
    "boolT",
    "i4T",
    "i8T",
    "i16T",
    "i32T",
    "i48T",
    "i64T",
    "bf16T",
    "fp16T",
    "fp32T",
    "fp8e4m3fnT",
    "fp8e5m2T",
    "fp8ue8m0T",
    "fp6e3m2fnT",
    "fp6e2m3fnT",
    "fp4e2m1fnT",
    "mxint8T",
    "bs32_fp8ue8m0_fp8e4m3fnT",
    "bs32_fp8ue8m0_fp8e5m2T",
    "bs32_fp8ue8m0_fp6e3m2fnT",
    "bs32_fp8ue8m0_fp6e2m3fnT",
    "bs32_fp8ue8m0_fp4e2m1fnT",
    "bs32_fp8ue8m0_mxint8T",
]

cond_list = [
    "CheckCondition::anyOf",
    "CheckCondition::allOf",
    "CheckCondition::invalid",
]


def extract_array_body(text: str, array_name: str) -> str:
    pattern = rf"\b{re.escape(array_name)}\[\]\s*=\s*\{{"
    match = re.search(pattern, text)
    if not match:
        raise RuntimeError(f"missing array {array_name}")
    pos = match.end()
    depth = 1
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    if depth != 0:
        raise RuntimeError("syntax error: curly bracket pair mismatch")
    return text[match.end() : pos - 1]


def split_top_level_braced_rows(body: str) -> list[str]:
    rows = []
    pos = 0
    while pos < len(body):
        while pos < len(body) and (body[pos].isspace() or body[pos] == ","):
            pos += 1
        if pos >= len(body):
            break
        if body[pos] != "{":
            raise RuntimeError("syntax error: expected '{' at start of row")
        start = pos + 1
        depth = 1
        pos += 1
        while pos < len(body) and depth > 0:
            if body[pos] == "{":
                depth += 1
            elif body[pos] == "}":
                depth -= 1
            pos += 1
        if depth != 0:
            raise RuntimeError("syntax error: curly bracket pair mismatch")
        rows.append(body[start : pos - 1].strip())
    return rows


def parse_array_ref(expr: str, expected_array: str, total_len: int) -> tuple[int, int]:
    match = re.fullmatch(
        rf"llvm::ArrayRef\(&{re.escape(expected_array)}\[(\d+)\],\s*(\d+)\)",
        expr.strip(),
    )
    if not match:
        raise RuntimeError(f"invalid ArrayRef expression: {expr}")
    offset = int(match.group(1))
    length = int(match.group(2))
    if length <= 0 or offset < 0 or offset + length > total_len:
        raise RuntimeError(
            f"out-of-bounds ArrayRef into {expected_array}: [{offset}, +{length})"
        )
    return offset, length


def verify_compliance_file_content(text: str) -> None:
    clean_text = re.sub(r"//[^\n]*", "", text)
    if clean_text.count("{") != clean_text.count("}"):
        raise RuntimeError("syntax error: curly bracket pair mismatch")

    type_body = extract_array_body(clean_text, "kComplianceTypeInfos")
    types = [t.strip() for t in type_body.split(",") if t.strip()]
    for ty in types:
        if ty not in type_list:
            raise RuntimeError(f"invalid type name {ty}")

    vti_rows = split_top_level_braced_rows(
        extract_array_body(clean_text, "kVersionedTypeInfos")
    )
    for row in vti_rows:
        parts = [p.strip() for p in row.rsplit(",", 1)]
        if len(parts) != 2:
            raise RuntimeError(f"invalid VersionedTypeInfo row: {row}")
        parse_array_ref(parts[0], "kComplianceTypeInfos", len(types))
        if not re.fullmatch(r"SpecificationVersion::V_\d+_\d+", parts[1]):
            raise RuntimeError(f"invalid specification version {parts[1]}")

    prof_modes = [
        m.strip()
        for m in extract_array_body(clean_text, "kProfileModes").split(",")
        if m.strip()
    ]
    for mode in prof_modes:
        if mode not in profile_list or not mode.startswith("Profile::"):
            raise RuntimeError(f"invalid profile name {mode}")

    ext_modes = [
        m.strip()
        for m in extract_array_body(clean_text, "kExtensionModes").split(",")
        if m.strip()
    ]
    for mode in ext_modes:
        if mode not in profile_list or not mode.startswith("Extension::"):
            raise RuntimeError(f"invalid profile name {mode}")

    def verify_info_array(
        array_name: str, mode_array_name: str, mode_count: int
    ) -> int:
        rows = split_top_level_braced_rows(extract_array_body(clean_text, array_name))
        for row in rows:
            fields = split_top_level_comma(row)
            if len(fields) not in (2, 3):
                raise RuntimeError(f"invalid {array_name} row: {row}")
            parse_array_ref(fields[0], mode_array_name, mode_count)
            if fields[1] != "{}":
                parse_array_ref(fields[1], "kVersionedTypeInfos", len(vti_rows))
            if len(fields) == 3 and fields[2] not in cond_list:
                raise RuntimeError(f"invalid condition name {fields[2]}")
        return len(rows)

    def verify_op_table(
        table_name: str, info_array_name: str, info_count: int
    ) -> None:
        rows = split_top_level_braced_rows(extract_array_body(clean_text, table_name))
        prev_op_name = None
        for row in rows:
            fields = split_top_level_comma(row)
            if len(fields) != 2:
                raise RuntimeError(f"invalid {table_name} row: {row}")
            op_match = re.fullmatch(r'"tosa\.([a-z0-9_]+)"', fields[0])
            if not op_match or op_match.group(1) not in op_list:
                raise RuntimeError(f"invalid tosa operation name {fields[0]}")
            op_name = op_match.group(1)
            if prev_op_name is not None and op_name <= prev_op_name:
                raise RuntimeError(f"unsorted operation table in {table_name}")
            prev_op_name = op_name
            parse_array_ref(fields[1], info_array_name, info_count)

    prof_info_count = verify_info_array(
        "kProfileComplianceInfos", "kProfileModes", len(prof_modes)
    )
    ext_info_count = verify_info_array(
        "kExtensionComplianceInfos", "kExtensionModes", len(ext_modes)
    )
    verify_op_table(
        "kProfileComplianceTable", "kProfileComplianceInfos", prof_info_count
    )
    verify_op_table(
        "kExtensionComplianceTable", "kExtensionComplianceInfos", ext_info_count
    )


def split_top_level_comma(text: str) -> list[str]:
    parts = []
    start = 0
    depth = 0
    for idx, ch in enumerate(text):
        if ch in "({":
            depth += 1
        elif ch in ")}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    parts.append(text[start:].strip())
    return parts


def make_valid_sample() -> str:
    return """
static constexpr TypeInfo kComplianceTypeInfos[] = {
    i8T, i32T,
};
static constexpr VersionedTypeInfo kVersionedTypeInfos[] = {
    {llvm::ArrayRef(&kComplianceTypeInfos[0], 2), SpecificationVersion::V_1_0},
};
static constexpr Profile kProfileModes[] = {
    Profile::pro_int,
};
static constexpr Extension kExtensionModes[] = {
    Extension::bf16, Extension::fp8e4m3fn,
};
static constexpr OpComplianceInfo<Profile> kProfileComplianceInfos[] = {
    {llvm::ArrayRef(&kProfileModes[0], 1), llvm::ArrayRef(&kVersionedTypeInfos[0], 1)},
};
static constexpr OpComplianceInfo<Extension> kExtensionComplianceInfos[] = {
    {llvm::ArrayRef(&kExtensionModes[0], 2), {}, CheckCondition::allOf},
};
static constexpr OpComplianceEntry<Profile> kProfileComplianceTable[] = {
    {"tosa.add", llvm::ArrayRef(&kProfileComplianceInfos[0], 1)},
};
static constexpr OpComplianceEntry<Extension> kExtensionComplianceTable[] = {
    {"tosa.dim", llvm::ArrayRef(&kExtensionComplianceInfos[0], 1)},
};
"""


def assert_raises(text: str, expected_msg: str) -> None:
    try:
        verify_compliance_file_content(text)
    except RuntimeError as e:
        assert expected_msg in str(e), f"Expected '{expected_msg}', got '{e}'"
        return
    raise AssertionError(f"Expected RuntimeError containing '{expected_msg}'")


def test_unknown_op():
    sample = make_valid_sample().replace('"tosa.add"', '"tosa.dummy"')
    assert_raises(sample, "invalid tosa operation name")


def test_unknown_prof():
    sample = make_valid_sample().replace("Profile::pro_int", "Profile::dummy")
    assert_raises(sample, "invalid profile name")


def test_unknown_extension():
    sample = make_valid_sample().replace("Extension::bf16", "Extension::dummy")
    assert_raises(sample, "invalid profile name")


def test_unknown_type():
    sample = make_valid_sample().replace("i8T", "i128T")
    assert_raises(sample, "invalid type name")


def test_unknown_condition():
    sample = make_valid_sample().replace(
        "CheckCondition::allOf", "CheckCondition::dummy"
    )
    assert_raises(sample, "invalid condition name")


def test_unknown_syntax():
    sample = make_valid_sample() + "}"
    assert_raises(sample, "curly bracket pair mismatch")


def test_unsorted_ops():
    sample = make_valid_sample().replace(
        '{"tosa.add", llvm::ArrayRef(&kProfileComplianceInfos[0], 1)},',
        '{"tosa.sub", llvm::ArrayRef(&kProfileComplianceInfos[0], 1)},\n'
        '    {"tosa.add", llvm::ArrayRef(&kProfileComplianceInfos[0], 1)},',
    )
    assert_raises(sample, "unsorted operation table")


def test_out_of_bounds_slice():
    sample = make_valid_sample().replace(
        "&kComplianceTypeInfos[0], 2", "&kComplianceTypeInfos[0], 5"
    )
    assert_raises(sample, "out-of-bounds ArrayRef")


def self_sanity_check() -> None:
    verify_compliance_file_content(make_valid_sample())
    test_unknown_op()
    test_unknown_prof()
    test_unknown_extension()
    test_unknown_type()
    test_unknown_condition()
    test_unknown_syntax()
    test_unsorted_ops()
    test_out_of_bounds_slice()


if __name__ == "__main__":
    import argparse

    self_sanity_check()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="Path to the generated compliance file"
    )
    args = parser.parse_args()

    with open(args.input, "r") as file:
        verify_compliance_file_content(file.read())
