#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <cmath>

using namespace std;

pair<vector<int>, unordered_map<int, int>> get_distributing2(int NUM_UCS, vector<int> y_pred) {
    unordered_map<int, int> tam_pools;
    for (int i = 0; i < y_pred.size(); i++) {
        tam_pools[y_pred[i]]++;
    }
    vector<pair<int, int>> tam_pools_vec(tam_pools.begin(), tam_pools.end());
    sort(tam_pools_vec.begin(), tam_pools_vec.end(), [](const pair<int, int>& a, const pair<int, int>& b) {
        return a.second > b.second;
    });
    int sum_clusters = y_pred.size();
    unordered_map<int, int> arredondamento;
    for (auto& p : tam_pools_vec) {
        arredondamento[p.first] = ceil(NUM_UCS * p.second / (double)sum_clusters);
    }
    int soma = 0;
    for (auto& p : arredondamento) {
        soma += p.second;
    }
    vector<pair<int, int>> xy(arredondamento.begin(), arredondamento.end());
    sort(xy.begin(), xy.end(), [](const pair<int, int>& a, const pair<int, int>& b) {
        return a.second < b.second;
    });
    while (soma != NUM_UCS) {
        for (auto& p : xy) {
            if (arredondamento[p.first] > 1) {
                arredondamento[p.first]--;
                break;
            }
        }
        vector<pair<int, int>> xx(arredondamento.begin(), arredondamento.end());
        sort(xx.begin(), xx.end(), [](const pair<int, int>& a, const pair<int, int>& b) {
            return a.second > b.second;
        });
        arredondamento.clear();
        for (auto& p : xx) {
            arredondamento[p.first] = p.second;
        }
        xy.assign(arredondamento.begin(), arredondamento.end());
        sort(xy.begin(), xy.end(), [](const pair<int, int>& a, const pair<int, int>& b) {
            return a.second < b.second;
        });
        soma = 0;
        for (auto& p : arredondamento) {
            soma += p.second;
        }
    }
    cout << "distribuicao final ";
    for (auto& p : arredondamento) {
        cout << p.first << ": " << p.second << " ";
    }
    cout << endl;
    
    vector<int> distribute;
    for (auto& p : arredondamento) {
        for (int j = 0; j < p.second; j++) {
            distribute.push_back(p.first);
        }
    }
    
    return make_pair(distribute, arredondamento);
}
