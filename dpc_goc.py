
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# ============================================================================
# 0. CHUẨN BỊ DỮ LIỆU MẪU (20 điểm từ Iris Dataset)
# ============================================================================
def lay_du_lieu():
    # 10 điểm nhóm 1 (Setosa) và 10 điểm nhóm 2 (Versicolor)
    X = np.array([
        [5.1, 3.5], [4.9, 3.0], [4.7, 3.2], [4.6, 3.1], [5.0, 3.6],
        [5.4, 3.9], [4.6, 3.4], [5.0, 3.4], [4.4, 2.9], [4.9, 3.1],
        [7.0, 3.2], [6.4, 3.2], [6.9, 3.1], [6.7, 3.1], [6.5, 2.8],
        [6.3, 2.5], [6.3, 3.3], [6.5, 3.0], [6.6, 2.9], [6.1, 2.9]
    ])
    return X

# ============================================================================
# 1. CÁC BƯỚC THỰC HIỆN THUẬT TOÁN DPC GỐC
# ============================================================================
def chay_thuat_toan_dpc_goc(X, k_clusters=2):
    n = len(X)
    
    # BƯỚC 1: Tính ma trận khoảng cách Euclidean giữa mọi cặp điểm
    D = cdist(X, X, metric='euclidean')
    
    # BƯỚC 2: Tính dc (cutoff distance) - thông số mật độ
    # Lấy giá trị ở mức 30% trong bảng khoảng cách để đảm bảo đủ độ bao phủ
    distances_flat = D[np.triu_indices(n, k=1)]
    dc = np.percentile(distances_flat, 30)
    
    # BƯỚC 3: Tính mật độ cục bộ (Rho) bằng Gaussian Kernel
    # Điểm càng đông người xung quanh thì Rho càng cao
    rho = np.zeros(n)
    for i in range(n):
        rho[i] = np.sum(np.exp(-(D[i, :] / dc) ** 2)) - 1
        
    # BƯỚC 4: Tính khoảng cách tách biệt (Delta)
    # Delta là khoảng cách từ điểm i đến điểm gần nhất có mật độ cao hơn nó
    delta = np.zeros(n)
    rho_sorted_indices = np.argsort(rho)[::-1] # Sắp xếp giảm dần mật độ
    
    # Điểm có mật độ cao nhất: Delta là khoảng cách lớn nhất trong tập dữ liệu
    delta[rho_sorted_indices[0]] = np.max(D[rho_sorted_indices[0], :])
    
    # Các điểm còn lại
    for i in range(1, n):
        current_idx = rho_sorted_indices[i]
        higher_density_indices = rho_sorted_indices[:i]
        delta[current_idx] = np.min(D[current_idx, higher_density_indices])
        
    # BƯỚC 5: Tính chỉ số Gamma = Rho * Delta (Phát hiện tâm cụm)
    gamma = rho * delta
    
    # BƯỚC 6: Chọn K điểm có Gamma cao nhất làm tâm cụm
    gamma_sorted_indices = np.argsort(gamma)[::-1]
    cluster_centers = gamma_sorted_indices[:k_clusters]
    
    # BƯỚC 7: Gán nhãn cụm cho các điểm
    labels = -1 * np.ones(n, dtype=int)
    # Gán nhãn cho các tâm trước
    for i in range(k_clusters):
        labels[cluster_centers[i]] = i
        
    # Gán các điểm còn lại dựa trên điểm "lân cận mạnh hơn"
    for i in rho_sorted_indices:
        if labels[i] == -1:
            higher_density_points = np.where(rho > rho[i])[0]
            if len(higher_density_points) > 0:
                nearest_higher_point = higher_density_points[np.argmin(D[i, higher_density_points])]
                labels[i] = labels[nearest_higher_point]
                
    return labels, cluster_centers

# ============================================================================
# 2. HIỂN THỊ KẾT QUẢ
# ============================================================================
if __name__ == "__main__":
    # Chuẩn bị
    X = lay_du_lieu()
    K = 3 
    
    # Chạy
    print(f"--- Dang chay DPC Goc voi K={K}  ---")
    nhan, tam = chay_thuat_toan_dpc_goc(X, k_clusters=K)
    
    print("Kết quả gán nhãn:", nhan)
    
    # Vẽ hình
    plt.figure(figsize=(8, 7))
    
    # Định nghĩa bảng màu giống trong hình: Xanh lá nhạt, Đỏ, Tím
    # Cụm 0: lightgreen, Cụm 1: red, Cụm 2: mediumpurple (hoặc blueviolet)
    custom_colors = ['#9aff9a', '#ff4500', '#9370db'] 
    
    for i in range(K):
        cluster_points = X[nhan == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                    c=custom_colors[i], s=100, edgecolors='black', label=f'Cụm {i+1}')
    
    # Vẽ tâm cụm bằng dấu X đen
    plt.scatter(X[tam, 0], X[tam, 1], c='black', marker='X', s=200, edgecolors='white', linewidth=1)
    
    plt.title("DPC Gốc", fontsize=14)
    plt.xlabel("") # Để trống nhãn trục giống hình
    plt.ylabel("")
    plt.grid(True, linestyle='-', alpha=0.3)
    plt.show()
