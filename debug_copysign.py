"""Diagnostic script for copysign float32 + int64 test failure."""
import torch
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")

# Test 1: basic copysign with float32 and int64
a = torch.tensor([3.0, -3.0, 0.0, -0.0], dtype=torch.float32)
b = torch.tensor([-5, 0, -5, -5], dtype=torch.int64)

print("\n=== Test 1: Basic copysign ===")
print(f"a = {a}")
print(f"b = {b}")

result = torch.copysign(a, b)
print(f"torch.copysign(a, b) = {result}")
print(f"result dtype: {result.dtype}")

# NumPy comparison
np_a = a.cpu().numpy()
np_b = b.cpu().numpy()
np_result = np.copysign(np_a, np_b)
print(f"np.copysign(np_a, np_b) = {np_result}")
print(f"np_result dtype: {np_result.dtype}")

print(f"\nPyTorch == NumPy (values): {torch.from_numpy(np_result).eq(result)}")

# Test 2: copysign with 0.0 magnitude and negative int64 sign
print("\n=== Test 2: copysign(0.0, negative int) ===")
a2 = torch.tensor([0.0, -0.0], dtype=torch.float32)
b2 = torch.tensor([-5, -5], dtype=torch.int64)
result2 = torch.copysign(a2, b2)
print(f"torch.copysign({a2}, {b2}) = {result2}")
print(f"  [0] raw bits: 0x{result2[0].view(torch.int32).item() & 0xFFFFFFFF:08X}")
print(f"  [1] raw bits: 0x{result2[1].view(torch.int32).item() & 0xFFFFFFFF:08X}")

np_a2 = a2.cpu().numpy()
np_b2 = b2.cpu().numpy()
np_result2 = np.copysign(np_a2, np_b2)
print(f"np.copysign result = {np_result2}")
np_f32 = np_result2.astype(np.float32)
print(f"  [0] raw bits: 0x{np_f32[0].view('<u4'):08X}")
print(f"  [1] raw bits: 0x{np_f32[1].view('<u4'):08X}")

# Test 3: sign check (like the test does)
print("\n=== Test 3: Sign check ===")
sign_check_torch = torch.copysign(torch.tensor(1.0), result2)
sign_check_np = torch.copysign(torch.tensor(1.0), torch.from_numpy(np_result2))
print(f"torch sign check: {sign_check_torch}")
print(f"numpy sign check: {sign_check_np}")
print(f"Match: {sign_check_torch.eq(sign_check_np)}")

# Test 4: broadcast case (like the failing test)
print("\n=== Test 4: Broadcast case (like the test) ===")
torch.manual_seed(42)
a4 = torch.tensor([0.0], dtype=torch.float32)
b4 = torch.randint(-9, 10, (10, 10), dtype=torch.int64)
result4 = torch.copysign(a4, b4)

np_a4 = a4.cpu().numpy()
np_b4 = b4.cpu().numpy()
np_result4 = np.copysign(np_a4, np_b4)
np_f32_4 = np_result4.astype(np.float32)

# Value check
value_match = torch.from_numpy(np_result4).eq(result4.to(torch.float64))
print(f"Value check (all equal): {value_match.all().item()}")

# Sign check
sign_torch = torch.copysign(torch.tensor(1.0), result4)
sign_np = torch.copysign(torch.tensor(1.0), torch.from_numpy(np_result4))
sign_match = sign_torch.eq(sign_np)
mismatch_count = (~sign_match).sum().item()
print(f"Sign check mismatches: {mismatch_count} / {sign_match.numel()}")
if mismatch_count > 0:
    mismatch_indices = (~sign_match).nonzero(as_tuple=False)
    print(f"First 5 mismatch indices: {mismatch_indices[:5].tolist()}")
    for idx in mismatch_indices[:3]:
        i, j = idx[0].item(), idx[1].item()
        print(f"  [{i},{j}]: b={b4[i,j].item()}, "
              f"torch_result={result4[i,j].item():+.1f} (bits=0x{result4[i,j].view(torch.int32).item() & 0xFFFFFFFF:08X}), "
              f"np_result={np_f32_4[i,j]:+.1f} (bits=0x{np_f32_4[i,j].view('<u4'):08X})")

# Test 5: all special cases (like the full test)
print("\n=== Test 5: All special cases ===")
cases = [0.0, -0.0, float("inf"), float("-inf"), float("nan")]
torch.manual_seed(123)
b5 = torch.randint(-9, 10, (10, 10), dtype=torch.int64)
np_b5 = b5.cpu().numpy()

for case in cases:
    a5 = torch.tensor([case], dtype=torch.float32)
    r5 = torch.copysign(a5, b5)

    np_a5 = a5.cpu().numpy()
    np_r5 = np.copysign(np_a5, np_b5)

    # Sign check
    s_torch = torch.copysign(torch.tensor(1.0), r5)
    s_np = torch.copysign(torch.tensor(1.0), torch.from_numpy(np_r5))
    mismatches = (~s_torch.eq(s_np)).sum().item()
    print(f"  case={case:>8}: sign mismatches = {mismatches} / {s_torch.numel()}")
