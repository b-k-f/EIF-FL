# Combining Client-Based Anomaly Detection and Federated Learning for Energy Forecasting in Smart Buildings

**Author:** Bouchra Fakher  
**Published in:** Springer's Cluster Computing Journal  
**Date:** 2025  
**DOI:** Still Under Review  

## Abstract

In today’s interconnected world, energy consumption forecasting faces challenges due to client-side anomalies in time-series data. Federated Learning (FL) offers a decentralized solution by enabling forecasts without accessing user data directly. However, the effectiveness of the global model can still decline if local anomalies go undetected or inappropriately managed. To address this, we propose our lightweight EIF-FL framework that employs unsupervised ensemble Anomaly Detection (AD) and cleaning datasets on client-side before applying the FL process. EIF-FL consists of two layers. The first layer is located on the client-side and employs Isolation Forest (IF) and Elliptic Envelope (EE) with a majority voting for AD. The second layer utilizes Long Short-Term Memory (LSTM) to forecast data in a federated manner on the server-side. Simulations on building energy consumption datasets demonstrate EIF-FL’s ability to improve AD metrics significantly such as accuracy, precision, recall, etc., and FL performance measures such as losses and errors.

# For citation :

@article{fakher2025combining,
  title={Combining client-based anomaly detection and federated learning for energy forecasting in smart buildings},
  author={Fakher, Bouchra and Brahmia, Mohamed el Amine and Bennis, Ismail and Abouaissa, Abdelhafid},
  journal={Cluster Computing},
  volume={28},
  number={16},
  pages={1058},
  year={2025},
  publisher={Springer}
}
