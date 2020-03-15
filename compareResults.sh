MODEL_OUT="$(python test_dqn_pong.py model.pth)"
MODEL_PRETRAINED_OUT="$(python test_dqn_pong.py model_pretrained.pth)"

if [ "$MODEL_OUT" == "$MODEL_PRETRAINED_OUT" ]; then
  echo "Both models output same results."
else
  echo "Models have different outputs."
fi
