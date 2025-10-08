set -e

# Read default values from terraform outputs in the parent directory
echo "Reading default values from Terraform outputs..."
export NODEPOOL_NAME=$(terraform -chdir=.. output -raw gpu_nodepool_name)
export NUM_NODES=$(kubectl get nodes -l 'cloud.google.com/gke-nodepool=mwy-llm-d-a3u-nodes' --no-headers |wc -l)
export GPUS_PER_NODE=$(terraform -chdir=.. output -raw gpus_per_node)

# Override default values with command-line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --nodepool-name)
            NODEPOOL_NAME="$2"
            shift 2
            ;;
        --num-nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --gpus-per-node)
            GPUS_PER_NODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Substitute variables and apply the manifest - TODO: fix substitution in kubectl create command
sed -e "s/__NODEPOOL_NAME__/${NODEPOOL_NAME}/g" \
    -e "s/__NUM_NODES__/${NUM_NODES}/g" \
    -e "s/__GPUS_PER_NODE__/${GPUS_PER_NODE}/g" \
    nccl-test.yaml.tmpl | kubectl apply -f -

# Get the generated job name
JOB_NAME=$(kubectl get job -o jsonpath='{.items[0].metadata.name}')

# Print instructions
cat <<EOF

NCCL test job submitted.

To check the status of the job, run:
kubectl describe job $JOB_NAME

To check the logs of the job, run:
kubectl logs -f -l job-name=$JOB_NAME

To delete the job, run:
kubectl delete job $JOB_NAME

EOF
