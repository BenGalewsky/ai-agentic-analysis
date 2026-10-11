# mlflow-stack

Helm chart that deploys an MLflow tracking server with:

- [`oci://ghcr.io/mlflow/charts/mlflow`](https://github.com/mlflow/mlflow) 3.17.0: the
  tracking server with basic auth, exposed through an nginx ingress at
  `mlflow2.servicex.ssl-hep.org`.
- [groundhog2k `postgres`](https://github.com/groundhog2k/helm-charts/tree/master/charts/postgres)
  1.6.8: Postgres 18 holding both the tracking/registry metadata and the
  basic-auth user database. It persists to a 10Gi `rook-ceph-block` PVC.
- The cluster's Ceph object store for artifacts, which the server proxies. The
  bucket is provisioned by the cluster admin.

The MLflow pod keeps no local state, so it doesn't need a PVC and updates
with a rolling deployment.

## Prerequisites

These must exist in the release namespace. The chart creates none of them:

| Name | Kind | Provided by | Contents |
| --- | --- | --- | --- |
| `mlflow` | Secret + ConfigMap | cluster admin (ObjectBucketClaim) | bucket credentials and details |
| `mlflow-db` | Secret | you | Postgres passwords and the backend store `uri` |
| `mlflow-auth-secret` | Secret | you | Flask `secret-key` |
| `mlflow-auth-config` | Secret | you | `basic_auth.ini` pointing at Postgres |

See [`examples/secrets.example.yaml`](examples/secrets.example.yaml) for the
expected keys. Encrypt the filled-in copy with SOPS before committing it
anywhere.

## Install

```bash
helm dependency build helm
helm upgrade --install mlflow helm -n <namespace>
```

The Postgres Service name (`mlflow-postgres`) and MLflow resource names
(`mlflow`) are pinned by `fullnameOverride`, because the URIs in the Secrets
hard-code the Service name.

## Notes

- Postgres creates the database and user only on first start with an empty
  data directory. Changing `mlflow-db` afterwards doesn't change the database.
  To rotate a password, run `ALTER USER` and then update both Secrets.
- `helm uninstall` leaves the Postgres PVC (`postgres-data-mlflow-postgres-0`)
  in place. Delete it by hand to start fresh.
- Nothing from the earlier SQLite-backed deployment is migrated, including
  experiments, runs, and basic-auth users.
