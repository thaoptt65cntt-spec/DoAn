
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# ============================================================================
# 0. CHUẨN BỊ DỮ LIỆU MẪU (20 điểm từ Iris Dataset)
# ============================================================================
def lay_du_lieu_mau():
    # 10 điểm nhóm 1 (Setosa) và 10 điểm nhóm 2 (Versicolor)
    # Dữ liệu chọn lọc để thấy rõ sự phân cụm
    X = np.array([
        [5.1, 3.5], [4.9, 3.0], [4.7, 3.2], [4.6, 3.1], [5.0, 3.6],
        [5.4, 3.9], [4.6, 3.4], [5.0, 3.4], [4.4, 2.9], [4.9, 3.1],
        [7.0, 3.2], [6.4, 3.2], [6.9, 3.1], [6.7, 3.1], [6.5, 2.8],
        [6.3, 2.5], [6.3, 3.3], [6.5, 3.0], [6.6, 2.9], [6.1, 2.9]
    ])
    return X

# ============================================================================
# 1. THUẬT TOÁN GB-DPC (TỰ ĐỘNG HÓA HOÀN TOÀN)
# ============================================================================
def chay_thuat_toan_gap_dpc(X):
    n = len(X)
    D = cdist(X, X, metric='euclidean')
    
    # --- PHẦN 1: Tính Rho và Delta giống bản gốc ---
    all_dists = D[np.triu_indices(n, k=1)]
    dc = np.percentile(all_dists, 30)
    
 # Tính mật độ cut-off
    rho = np.zeros(n)
    for i in range(n):
        rho[i] = np.sum(D[i, :] < dc) - 1
        
    delta = np.zeros(n)
    rho_sort = np.argsort(rho)[::-1]
    delta[rho_sort[0]] = np.max(D[rho_sort[0], :])
    for i in range(1, n):
        idx = rho_sort[i]
        higher = rho_sort[:i]
        delta[idx] = np.min(D[idx, higher])
        
    # --- PHẦN 2: TỰ ĐỘNG TÌM K (Bản chất của Gap-DPC) ---
    gamma = rho * delta
    gamma_sort_indices = np.argsort(gamma)[::-1]
    gamma_values_sorted = gamma[gamma_sort_indices]
    
    # Tính Gap (Độ tụt giá trị) giữa các điểm
    # Gap = Gamma[i] - Gamma[i+1]
    gaps = -np.diff(gamma_values_sorted)
    
    # Tìm xem Gap nào lớn nhất (thường nằm ở top 10 điểm đầu tiên)
    # Vị trí Gap lớn nhất sẽ cho biết ranh giới giữa tâm cụm và các điểm thường
    vi_tri_max_gap = np.argmax(gaps[:10]) 
    so_cum_tu_dong = vi_tri_max_gap + 1
    
    # Các tâm cụm là những điểm nằm trên Gap này
    tam_cum = gamma_sort_indices[:so_cum_tu_dong]
    
    # --- PHẦN 3: Gán nhãn cho các điểm ---
    nhan = -1 * np.ones(n, dtype=int)
    for i in range(so_cum_tu_dong):
        nhan[tam_cum[i]] = i
        
    # Lặp để gán nhãn (Đảm bảo tất cả được gán thông qua chuỗi liên kết)
    while -1 in nhan:
        da_gan_them = False
        for i in range(n):
            if nhan[i] == -1:
                diem_cao_hon = np.where(rho > rho[i])[0]
                if len(diem_cao_hon) > 0:
                    gan_nhat = diem_cao_hon[np.argmin(D[i, diem_cao_hon])]
                    if nhan[gan_nhat] != -1:
                        nhan[i] = nhan[gan_nhat]
                        da_gan_them = True
        if not da_gan_them: break # Đề phòng dữ liệu nhiễu
        
    return nhan, tam_cum, so_cum_tu_dong

# ============================================================================
# 2. HIỂN THỊ KẾT QUẢ
# ============================================================================
if __name__ == "__main__":
    X = lay_du_lieu_mau()
    
    print("--- Đang chạy GB-DPC Cải tiến (Tự động phát hiện K) ---")
    nhan, tam, K_ket_qua = chay_thuat_toan_gap_dpc(X)
    
    print(f"Thuật toán đã tự tìm thấy K = {K_ket_qua} cụm.")
    print("Kết quả gán nhãn:", nhan)
    
    # Vẽ hình
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:,0], X[:,1], c=nhan, cmap='tab10', s=100)
    plt.scatter(X[tam, 0], X[tam, 1], c='red', marker='*', s=300, label='Tâm cụm tự động')
    plt.title(f"Kết quả GB-DPC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
