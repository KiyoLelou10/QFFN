"""Numerical diagnostics accompanying qfnn_chebyshev_revised_v6.tex.

These checks do not replace the analytic proofs.  They test the identities
and small-dimensional ranks used in Appendix D of the manuscript.
"""

from __future__ import annotations

import itertools
import math
import numpy as np


RNG = np.random.default_rng(20260811)


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def chebyshev_t(degree: int, z):
    if degree == 0:
        return np.ones_like(z)
    if degree == 1:
        return z
    t0 = np.ones_like(z)
    t1 = z
    for _ in range(2, degree + 1):
        t0, t1 = t1, 2 * z * t1 - t0
    return t1


def random_projector(dim: int, rank: int) -> np.ndarray:
    x = RNG.normal(size=(dim, rank)) + 1j * RNG.normal(size=(dim, rank))
    q, _ = np.linalg.qr(x)
    return q @ q.conj().T


def check_reflection_identity() -> float:
    worst = 0.0
    for dim in (2, 3, 5):
        for rank in range(1, dim):
            for degree in range(0, 9):
                for _ in range(12):
                    psi = normalize(RNG.normal(size=dim) + 1j * RNG.normal(size=dim))
                    p_state = np.outer(psi, psi.conj())
                    q = random_projector(dim, rank)
                    r_psi = np.eye(dim) - 2 * p_state
                    r_q = np.eye(dim) - 2 * q
                    lhs = np.vdot(psi, np.linalg.matrix_power(r_psi @ r_q, degree) @ psi)
                    overlap = float(np.real(np.vdot(psi, q @ psi)))
                    rhs = chebyshev_t(degree, 2 * overlap - 1)
                    worst = max(worst, abs(lhs - rhs))
    assert worst < 5e-12, worst
    return worst


def compositions(total: int, length: int):
    if length == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for tail in compositions(total - first, length - 1):
            yield (first,) + tail


def coherent_symmetric_vector(v: np.ndarray, degree: int) -> np.ndarray:
    coords = []
    d_fact = math.factorial(degree)
    for mu in compositions(degree, len(v)):
        multinomial = d_fact
        monomial = 1.0 + 0.0j
        for vi, mi in zip(v, mu):
            multinomial /= math.factorial(mi)
            monomial *= vi ** mi
        coords.append(math.sqrt(multinomial) * monomial)
    return np.asarray(coords, dtype=complex)


def hermitian_real_vector(a: np.ndarray) -> np.ndarray:
    # Redundant full real/imaginary vectorization is harmless for real rank.
    return np.concatenate([a.real.ravel(), a.imag.ravel()])


def check_coherent_span(n: int, degree: int) -> int:
    sym_dim = math.comb(n + degree - 1, degree)
    target = sym_dim**2
    vectors = []
    for _ in range(5 * target):
        v = normalize(RNG.normal(size=n) + 1j * RNG.normal(size=n))
        w = coherent_symmetric_vector(v, degree)
        vectors.append(hermitian_real_vector(np.outer(w, w.conj())))
    rank = int(np.linalg.matrix_rank(np.stack(vectors), tol=1e-9))
    assert rank == target, (n, degree, rank, target)
    return rank


def check_interpolation() -> float:
    m, dim = 7, 4
    states = []
    for _ in range(m):
        psi = normalize(RNG.normal(size=dim) + 1j * RNG.normal(size=dim))
        states.append(np.outer(psi, psi.conj()))
    labels = RNG.normal(size=m)

    def g(i: int, j: int, rho: np.ndarray) -> float:
        a = states[i] - states[j]
        delta = np.trace(a @ a).real
        return float((np.trace(a @ rho) - np.trace(a @ states[j])).real / delta)

    predicted = []
    for rho in states:
        value = 0.0
        for i in range(m):
            basis = 1.0
            for j in range(m):
                if i != j:
                    basis *= g(i, j, rho)
            value += labels[i] * basis
        predicted.append(value)
    error = float(np.max(np.abs(np.asarray(predicted) - labels)))
    assert error < 2e-11, error
    return error


def check_alternating_separation(degree: int) -> tuple[int, int]:
    labels = np.asarray([1 if k % 2 == 0 else -1 for k in range(2 * degree)])
    n = len(labels)
    best_errors = n
    # A sinusoid plus bias has a positive set that is empty, full, or one cyclic block.
    candidates = [np.full(n, -1), np.full(n, 1)]
    for start in range(n):
        for length in range(1, n):
            pred = np.full(n, -1)
            for offset in range(length):
                pred[(start + offset) % n] = 1
            candidates.append(pred)
    for pred in candidates:
        best_errors = min(best_errors, int(np.count_nonzero(pred != labels)))

    theta = np.arange(n) * math.pi / degree
    qfnn = np.sign(np.cos(degree * theta)).astype(int)
    qfnn_errors = int(np.count_nonzero(qfnn != labels))
    assert best_errors == degree - 1, (degree, best_errors)
    assert qfnn_errors == 0, (degree, qfnn_errors)
    return best_errors, qfnn_errors


def check_scalar_routing() -> float:
    worst = 0.0
    for degrees in ((2, 3), (2, 2, 3), (3, 4, 2)):
        for z in np.linspace(-1.0, 1.0, 1001):
            routed = z
            for degree in degrees:
                routed = chebyshev_t(degree, routed)
            direct = chebyshev_t(math.prod(degrees), z)
            worst = max(worst, abs(routed - direct))
    assert worst < 2e-10, worst
    return worst


def check_effective_dimension_monotonicity() -> float:
    """Stress-test the pure-state K_D = K_1**D monotonicity proposition."""
    smallest_increment = float("inf")
    for sample_size, dim in ((6, 2), (9, 3), (12, 5)):
        for _ in range(40):
            states = np.stack(
                [
                    normalize(RNG.normal(size=dim) + 1j * RNG.normal(size=dim))
                    for _ in range(sample_size)
                ]
            )
            k1 = np.abs(states @ states.conj().T) ** 2
            for ridge in (1e-4, 1e-2, 1.0):
                previous = None
                for degree in range(1, 8):
                    kd = k1**degree
                    eigenvalues = np.linalg.eigvalsh(kd).clip(min=0.0)
                    effective = float(
                        np.sum(eigenvalues / (eigenvalues + sample_size * ridge))
                    )
                    if previous is not None:
                        increment = effective - previous
                        smallest_increment = min(smallest_increment, increment)
                        assert increment > -2e-10, (
                            sample_size,
                            dim,
                            ridge,
                            degree,
                            increment,
                        )
                    previous = effective
    return smallest_increment


def check_diagonal_head_compilation() -> float:
    """Check the exact finite-bank identity in Corollary 10.7."""
    worst = 0.0
    for output_dim in (2, 4, 8):
        for _ in range(100):
            widths = RNG.integers(1, 7, size=output_dim)
            coefficients = [RNG.normal(size=int(width)) for width in widths]
            # Exercise the constant-coordinate branch of the proof.
            coefficients[0] = np.zeros_like(coefficients[0])
            responses = [RNG.uniform(-1.0, 1.0, size=int(width)) for width in widths]
            biases = RNG.normal(size=output_dim)

            s_by_class = np.asarray([np.sum(np.abs(a)) for a in coefficients])
            total = float(np.sum(s_by_class))
            expected = np.asarray(
                [
                    bias + float(np.dot(a, alpha))
                    for bias, a, alpha in zip(biases, coefficients, responses)
                ]
            )

            if total == 0.0:
                recovered = biases.copy()
            else:
                deltas = []
                for class_norm, a, alpha in zip(
                    s_by_class, coefficients, responses
                ):
                    if class_norm == 0.0:
                        # P_c = 0 gives p_c = 0 and delta_c = -1.
                        deltas.append(-1.0)
                        continue
                    weights = np.abs(a) / total
                    signs = np.sign(a)
                    overlap = float(np.sum(weights * (1.0 + signs * alpha) / 2.0))
                    deltas.append(2.0 * overlap - 1.0)
                deltas = np.asarray(deltas)
                shifted_biases = biases + total - s_by_class
                recovered = total * deltas + shifted_biases

            worst = max(worst, float(np.max(np.abs(recovered - expected))))

    assert worst < 5e-12, worst
    return worst


def main() -> None:
    reflection_error = check_reflection_identity()
    ranks = {
        (2, 2): check_coherent_span(2, 2),
        (2, 3): check_coherent_span(2, 3),
        (3, 2): check_coherent_span(3, 2),
    }
    interpolation_error = check_interpolation()
    separations = {d: check_alternating_separation(d) for d in (2, 3, 4, 7)}
    routing_error = check_scalar_routing()
    effective_dimension_increment = check_effective_dimension_monotonicity()
    diagonal_head_error = check_diagonal_head_compilation()

    print("QFNN theorem sanity checks: PASS")
    print(f"projector-reflection maximum error: {reflection_error:.3e}")
    print(f"coherent-operator real ranks: {ranks}")
    print(f"interpolation maximum error: {interpolation_error:.3e}")
    print(f"alternating separation (best affine errors, QFNN errors): {separations}")
    print(f"scalar-routing composition maximum error: {routing_error:.3e}")
    print(
        "smallest observed effective-dimension increment: "
        f"{effective_dimension_increment:.3e}"
    )
    print(f"diagonal-head compilation maximum error: {diagonal_head_error:.3e}")


if __name__ == "__main__":
    main()
