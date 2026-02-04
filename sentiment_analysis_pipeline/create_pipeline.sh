
if [ ! -d "pipelines/sentiment_analysis_workflow" ]; then
  mkdir -p pipelines/sentiment_analysis_workflow
  cat > pipelines/sentiment_analysis_workflow/metadata.yaml <<EOF
blocks: []
callbacks: []
concurrency_config: {}
conditionals: []
created_at: '2026-02-02 06:00:00'
data_integration: null
description: End-to-end Sentiment Analysis Pipeline
executor_config: {}
executor_count: 1
executor_type: null
extensions: {}
name: sentiment_analysis_workflow
notification_config: {}
retry_config: {}
run_pipeline_in_one_process: false
settings:
  triggers: null
spark_config: {}
tags: []
type: python
updated_at: '2026-02-02 06:00:00'
uuid: sentiment_analysis_workflow
variables_dir: /home/rahul/.mage_data/sentiment_analysis_pipeline
widgets: []
EOF
  echo "Created pipeline sentiment_analysis_workflow manually."
else
  echo "Pipeline already exists."
fi
