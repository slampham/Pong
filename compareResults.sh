MODEL_FILENAME="models/model.pth"
MODEL_PRETRAINED_FILENAME="/models/model_pretrained.pth"

MODEL_OUT="$(python test_dqn_pong.py $MODEL_FILENAME)"
MODEL_PRETRAINED_OUT="$(python test_dqn_pong.py $MODEL_PRETRAINED_FILENAME)"

if [ "$MODEL_OUT" == "$MODEL_PRETRAINED_OUT" ]; then
  echo "Both models output same results."
else
  echo "Models have different outputs."
fi
