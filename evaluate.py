def precision_at_k(relevant, retrieved, k):
    retrieved_k = retrieved[:k]
    hits = len(set(retrieved_k) & set(relevant))
    return hits / k


if __name__ == "__main__":

    relevant = [1, 2, 3]
    retrieved = [1, 5, 2, 7, 9]

    print("Precision@3 =", precision_at_k(relevant, retrieved, 3))