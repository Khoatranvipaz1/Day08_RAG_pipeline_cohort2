# Tai lieu mau ve danh muc chat ma tuy

- Source: LabDataset
- Year: 2024
- Type: legal

Tai lieu mau ve danh muc chat ma tuy, tien chat va cac chat huong than phuc vu lab RAG Day 08.

Noi dung nay khong thay the van ban phap luat chinh thuc. Muc dich cua file la bo sung corpus de kiem tra retrieval, BM25, hybrid search, citation va abstention. Tai lieu mo ta cach he thong RAG can xu ly cac cau hoi ve danh muc chat ma tuy, tien chat, kiem soat dac biet, nguon du lieu va bang chung trong context.

Khi nguoi dung hoi ve danh muc chat ma tuy, retriever can lay dung chunk co tu khoa "danh muc", "chat ma tuy", "tien chat", "chat huong than" va "kiem soat dac biet". Dense search co the bat y nghia tong quat ve nhom chat can quan ly, trong khi BM25 giup bat exact keyword va ten nhom chat. Hybrid search bang RRF ket hop hai nguon de tang context recall.

Trong generation, assistant phai noi ro neu thong tin chi den tu tai lieu mau. Neu cau hoi yeu cau ket luan chinh thuc ve mot chat cu the ma context khong neu, assistant phai tra "I cannot verify this information" thay vi tu suy dien. Cau tra loi can co citation den source LabDataset hoac chunk id tuong ung.
