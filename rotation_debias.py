"""
Rotation-based debiasing for MedCPT query embeddings.

Applies rotation-based debiasing to query vectors before FAISS retrieval,
reducing demographic bias without amplifying bias in unconsidered concepts.

Core formula:
    v_rot = v - P_C·v + ||P_C·v|| · w

where P_C is the projection onto the demographic subspace and w is a
neutral vector with equal cosine similarity to all demographic groups.

Usage:
    # Build debiaser (offline, once)
    debiaser = RotationDebiaser.from_medcpt(encoder_path, concepts=["race", "gender"])
    debiaser.save("rotation_debiaser.npz")

    # Apply at inference
    debiaser = RotationDebiaser.load("rotation_debiaser.npz")
    xq_debiased = debiaser.debias_batch(xq)
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "retriever"))


GROUP_TEXTS = {
    "race": {
        "caucasian":        ["Caucasian patient", "Caucasian man", "Caucasian woman",
                             "white patient", "white man", "white woman", "a Caucasian"],
        "african_american": ["African American patient", "African American man",
                             "African American woman", "Black patient", "Black man",
                             "Black woman", "an African American"],
        "hispanic":         ["Hispanic patient", "Hispanic man", "Hispanic woman",
                             "Latino patient", "Latina patient", "a Hispanic"],
        "asian":            ["Asian patient", "Asian man", "Asian woman", "an Asian"],
        "native_american":  ["Native American patient", "Native American man",
                             "Native American woman", "Indigenous patient",
                             "a Native American"],
    },
    "gender": {
        "male":   ["a man presents", "a male patient", "year-old man",
                   "year-old male", "he presents", "his symptoms"],
        "female": ["a woman presents", "a female patient", "year-old woman",
                   "year-old female", "she presents", "her symptoms"],
    },
}


class RotationDebiaser:
    """
    Rotation-based demographic debiaser for dense query embeddings.

    Attributes:
        P : dict[concept -> (d, d) ndarray]  projection matrices
        w : dict[concept -> (d,) ndarray]    neutral direction vectors
        A : dict[concept -> (d, n) ndarray]  group direction matrices
    """

    def __init__(self):
        self.P = {}
        self.w = {}
        self.A = {}
        self._P_joint = None
        self._w_joint = None

    def build_joint(self):
        """Build joint projection matrix over all registered concepts."""
        A_joint = np.hstack(list(self.A.values()))
        AtA_inv = np.linalg.pinv(A_joint.T @ A_joint)
        self._P_joint = A_joint @ AtA_inv @ A_joint.T
        ones = np.ones(A_joint.shape[1])
        w_tilde = A_joint @ AtA_inv @ ones
        self._w_joint = w_tilde / (np.linalg.norm(w_tilde) + 1e-12)
        print(f"Built joint rotation matrix (concepts: {list(self.A.keys())}, "
              f"subspace dim: {A_joint.shape[1]})")

    def add_concept(self, concept: str, group_vecs: np.ndarray):
        """Register a demographic concept for debiasing.

        Args:
            concept    : e.g. "race"
            group_vecs : (n_groups, d) array of group embeddings
        """
        A = group_vecs.T.astype(np.float64)
        AtA_inv = np.linalg.pinv(A.T @ A)
        self.A[concept] = A
        self.P[concept] = A @ AtA_inv @ A.T
        ones = np.ones(A.shape[1])
        w_tilde = A @ AtA_inv @ ones
        self.w[concept] = w_tilde / (np.linalg.norm(w_tilde) + 1e-12)

    def debias_batch(self, Q: np.ndarray, mode: str = "joint") -> np.ndarray:
        """Apply rotation debiasing to a batch of query embeddings.

        Args:
            Q    : (N, d) query embedding matrix
            mode : 'joint' (default) or 'sequential'

        Returns:
            (N, d) debiased embeddings
        """
        V = Q.astype(np.float64)
        if mode == "joint":
            if self._P_joint is None:
                self.build_joint()
            P_CV = V @ self._P_joint.T
            norms = np.linalg.norm(P_CV, axis=1, keepdims=True)
            V = V - P_CV + norms * self._w_joint[np.newaxis, :]
        else:
            for c in self.P:
                P_CV = V @ self.P[c].T
                norms = np.linalg.norm(P_CV, axis=1, keepdims=True)
                V = V - P_CV + norms * self.w[c][np.newaxis, :]
        return V.astype(np.float32)

    def debias(self, q: np.ndarray, mode: str = "joint") -> np.ndarray:
        return self.debias_batch(q[np.newaxis], mode=mode)[0]

    def save(self, path: str):
        data = {}
        for c in self.P:
            data[f"P_{c}"] = self.P[c]
            data[f"w_{c}"] = self.w[c]
            data[f"A_{c}"] = self.A[c]
        np.savez(path, **data)
        print(f"Saved debiaser to {path}  (concepts: {list(self.P.keys())})")

    @classmethod
    def load(cls, path: str):
        obj = cls()
        data = np.load(path)
        concepts = set(k.split("_", 1)[1] for k in data.files)
        for c in concepts:
            obj.P[c] = data[f"P_{c}"]
            obj.w[c] = data[f"w_{c}"]
            obj.A[c] = data[f"A_{c}"]
        print(f"Loaded debiaser from {path}  (concepts: {list(obj.P.keys())})")
        return obj

    @classmethod
    def from_medcpt(cls, encoder_path: str, concepts=None, group_texts=None):
        """Build rotation debiaser by encoding group texts with MedCPT."""
        import query_encode as qe

        if group_texts is None:
            group_texts = GROUP_TEXTS
        if concepts is None:
            concepts = list(group_texts.keys())

        obj = cls()
        for concept in concepts:
            groups = group_texts[concept]
            group_vecs = []
            print(f"\nEncoding group directions for concept: {concept}")
            for gname, texts in groups.items():
                vecs = qe.query_encode(texts, model_path=encoder_path)
                avg = vecs.mean(axis=0)
                avg = avg / (np.linalg.norm(avg) + 1e-12)
                group_vecs.append(avg)
                print(f"  {gname}: {len(texts)} phrasings")
            obj.add_concept(concept, np.stack(group_vecs))
        return obj


def main():
    parser = argparse.ArgumentParser(description="Build rotation debiaser from MedCPT embeddings")
    parser.add_argument("--encoder_path", required=True)
    parser.add_argument("--output_path", default="rotation_debiaser.npz")
    parser.add_argument("--concepts", nargs="+", default=["race", "gender"])
    args = parser.parse_args()

    debiaser = RotationDebiaser.from_medcpt(args.encoder_path, concepts=args.concepts)
    debiaser.save(args.output_path)


if __name__ == "__main__":
    main()
