k8s pods for online_inference app

* probes.yaml checks the liveliness and readiness of the app, so the original app was modified to shut down
  after one minute and starts with a 20 seconds delay
