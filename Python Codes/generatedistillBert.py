import tensorflow as tf
from transformers import TFDistilBertForQuestionAnswering, DistilBertTokenizer

# Load model and tokenizer
model_name = 'distilbert-base-uncased-distilled-squad'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = TFDistilBertForQuestionAnswering.from_pretrained(model_name)

# Create a wrapper to handle the input format differences
class DistilBERTQAWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    # Define inputs to match your Java implementation
    @tf.function(input_signature=[
        tf.TensorSpec([1, 384], tf.float32, name='input_ids'),
        tf.TensorSpec([1, 384], tf.float32, name='attention_mask'),
        tf.TensorSpec([1, 384], tf.float32, name='segment_ids')  # Unused but included for compatibility
    ])
    def call(self, input_ids, attention_mask, segment_ids):
        # Convert float inputs to int32 (as required by DistilBERT)
        input_ids_int = tf.cast(input_ids, tf.int32)
        attention_mask_int = tf.cast(attention_mask, tf.int32)
        
        # Run the model (ignore segment_ids)
        outputs = self.model(
            input_ids=input_ids_int,
            attention_mask=attention_mask_int
        )
        
        # Return outputs in expected format
        return {
            'start_logits': outputs.start_logits,
            'end_logits': outputs.end_logits
        }

# Wrap the model to handle input/output format differences
wrapped_model = DistilBERTQAWrapper(model)

# Convert to SavedModel format first
tf.saved_model.save(wrapped_model, "distilbert_saved_model")

# Convert SavedModel to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("distilbert_saved_model")
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

# Save the model
with open("distilbert-squad-384.tflite", "wb") as f:
    f.write(tflite_model)