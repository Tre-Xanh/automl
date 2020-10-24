---
marp: true
paginate: true
---

# MLflowでらくらく前処理・機械学習・予測API一連化

MLflow で実験のパラメータ、メトリックや学習済みモデルの記録については、情報が豊富に見つかりますが、しかし前処理と学習・予測を一連の処理としてパイプライン化する、分かりやすい簡単なサンプルが見つからなかったので、作ってみました。

よって、このサンプルでは、前処理と学習・予測のパイプライン化に重点を置きます。
また、MLflowで簡単に予測サービス(REST)を立ち上げられることについても少し触れます。

ソースコードはこちら
https://github.com/vochicong/automl

---

# 概要

* データ: [タイタニック号乗客の生存](https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv)
* [Pandas](https://pandas.pydata.org/)で前処理
* [H2O AutoML](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html?highlight=automl)で機械学習
* [MLflow custom model](https://mlflow.org/docs/latest/models.html#example-saving-an-xgboost-model-in-mlflow-format)で前処理と学習・予測をつなげる :notes:
* [MLflow models serve](https://mlflow.org/docs/latest/models.html#deploy-mlflow-models)で予測RESTサービス

---

# 開発環境の準備

[Conda](https://docs.conda.io/en/latest/miniconda.html)で環境作成

``` bash
make devenv
```

---

# コード

`main.py` にコードが全部入っています。

* Preproc: データ前処理のクラス
  + Age, Fareの Min-Maxスケーリング
* h2o_fit: H2OAutoMLで学習
* H2OPredictor: H2OAutoMLで予測

前処理・機械学習・テスト予測の実行

``` bash
make train
```

---

# 予測APIサービスの起動

デフォルトで5000番ポートが使われます。

``` bash
make serve
```

## 予測API用Dockerイメージ

簡単に作れます

``` bash
mlflow models build-docker -m $MLFLOW_MODEL
```

---

# 予測APIテスト

``` bash
make test
```

同じテストデータに対して、APIを使って予測させる場合と、
`main.py` でモデルを直接ロードして予測させる場合とを比較して、
同じ予測結果になることを確認します

---

# 参考）APIのJSON形式

Request

``` json
{ "columns": [ "x1", "x2", "x3" ],
  "data": [
               [ 3,    2,    5 ],
               [ 1,    4,    8 ] ] }
```

Response（分類問題）

``` json
[ { "predict": 0, "p0": 0.7, "p1": 0.3 },
  { "predict": 1, "p0": 0.6, "p1": 0.4 } ]
```
