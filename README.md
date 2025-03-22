# mla-c01-study-notes

## Important Links

- **📝 Official Exam Guide (PDF)**  
  [AWS Certified Machine Learning Engineer – Associate (MLA-C01) Official Guide](https://d1.awsstatic.com/training-and-certification/docs-ml/ml-engineer-associate-exam-guide.pdf)

- **📚 AWS Certified Machine Learning Engineer – Associate Page**  
  [Exam Overview, Registration, Sample Questions](https://aws.amazon.com/certification/certified-machine-learning-engineer-associate/)

- **🔧 SageMaker Developer Guide**  
  [Amazon SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)

- **📊 AWS Well-Architected ML Lens**  
  [Machine Learning Lens Whitepaper](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/welcome.html)


## Top 40 General Tips for MLA-C01 (Ordered by Importance)

1. **Use Managed AWS Services**: Always default to managed options like SageMaker, Glue, Bedrock unless you need full control.

2. **Minimize Operational Overhead**: Pick serverless (Lambda, Firehose, Glue) when possible for low-maintenance solutions.

3. **SageMaker Model Monitor = Detect Drift**: Use it to track feature distribution, data quality, and prediction drift.

4. **SageMaker Clarify = Detect Bias**: Use it pre- and post-deployment to ensure fairness and explainability.

5. **Hyperparameter Tuning = Use SageMaker AMT**: Automatic Model Tuning is the go-to for optimizing training jobs.

6. **Real-Time Inference = SageMaker Endpoints**: Use real-time endpoints for synchronous, low-latency predictions.

7. **Batch Inference = SageMaker Batch Transform**: Ideal for large volumes of data processed asynchronously.

8. **Async Inference = SageMaker Async Endpoints**: For large or slow inference requests with predictable duration.

9. **SageMaker Pipelines = Full MLOps Automation**: Preferred tool for building, training, evaluating, and deploying models.

10. **Use Spot Instances for Training**: Cuts cost significantly for non-critical or retryable jobs.

11. **Multi-AZ = High Availability**: Choose multi-AZ designs for fault tolerance in production workloads.

12. **Multi-Region = Disaster Recovery**: Use for business continuity and compliance across geographies.

13. **Minimize Overfitting = Use Regularization**: Techniques like L1, L2, dropout, and early stopping reduce model overfitting.

14. **Prefer Foundation Models for NLP = Use Bedrock**: For GenAI or NLP tasks, leverage pre-trained models via Bedrock.

15. **Low-Frequency Inference = Use Serverless Endpoints**: Cost-effective for occasional inference workloads.

16. **Model Reuse = SageMaker Feature Store**: Store and reuse features across multiple training jobs.

17. **CI/CD = CodePipeline + SageMaker or Step Functions**: Automate ML workflows from source to deploy.

18. **Edge Deployment = Use SageMaker Neo**: Compile and optimize models for deployment on edge devices.

19. **Secure Access = Use IAM with Least Privilege**: Always restrict roles and policies to just what's needed.

20. **Secure Endpoints = Deploy in VPC**: Isolate endpoints and training jobs inside a secure VPC.

21. **Scaling = Use Auto Scaling for Endpoints**: Configure invocations per instance or latency-based scaling.

22. **Monitor Everything = CloudWatch Logs, Metrics, Alarms**: Essential for debugging and tracking behavior.

23. **Audit Everything = Use CloudTrail**: Log all actions and events for security and traceability.

24. **Right-Size Inference = Use Inference Recommender**: Helps choose the best instance type based on latency and cost.

25. **Right-Size Training = Use Compute Optimizer**: Optimize training instances using usage data.

26. **Load Testing = Use Shadow Deployments**: Compare new models to production without customer impact.

27. **Trigger Workflows = Use EventBridge**: Automate pipelines or retraining when upstream events occur.

28. **Track Experiments = Use SageMaker Experiments**: Manage and compare model training runs and metadata.

29. **Secure CI/CD = IAM + Pipeline Encryption**: Apply access controls and encryption to protect pipelines.

30. **Join S3 + Redshift = Use Redshift Spectrum**: Query S3 directly using Redshift SQL.

31. **Low Latency Streaming = Use Kinesis Streams**: Best for sub-second processing needs.

32. **Near Real-Time = Use Kinesis Firehose**: Easier ingestion to S3/Redshift with minimal config.

33. **Fast File Transfer = Use S3 Transfer Acceleration**: Optimize uploads/downloads for global teams.

34. **Automate Infrastructure = Use CloudFormation or CDK**: Ensure reproducibility and CI/CD of infrastructure.

35. **No-Code ETL = Use DataBrew**: Quick transformations for non-engineers or prototyping.

36. **Rule-Based Data Checks = Use Glue Data Quality**: Validate schema, nulls, custom thresholds.

37. **Split Data Properly = Shuffle + Stratify**: Prevent data leakage and ensure balanced train/test sets.

38. **Explainability = Use SHAP with Clarify**: Interpret feature influence for compliance or trust.

39. **Anomaly Detection = Use Lookout for Metrics**: Automatically flag metric anomalies in production.

40. **Labeling = Use Ground Truth or Mechanical Turk**: For building high-quality labeled datasets at scale.



## Domain 1: Data Preparation for Machine Learning (ML)

---

### Task 1.1: Ingest and Store Data

#### Knowledge Of:
- Data Formats: Parquet, JSON, CSV, ORC, Avro, RecordIO  
  - Parquet/ORC = efficient columnar formats → best for analytics  
  - RecordIO = used in SageMaker training

- AWS Storage Services:
  - Amazon S3: Object store, scalable, cost-effective  
  - Amazon EFS: Scalable file system for Linux-based workloads  
  - FSx for NetApp ONTAP: Shared storage for enterprise apps

- Streaming Sources:
  - Kinesis Data Streams / Firehose  
  - Amazon MSK (Managed Streaming for Kafka)  
  - Amazon Flink (via Kinesis Data Analytics) for real-time processing

#### Skills In:
- Extracting Data from AWS Sources:
  - Use S3 Transfer Acceleration for faster global uploads
  - Use EBS Provisioned IOPS for high-performance access

- Choosing Formats by Access Pattern:
  - Columnar (Parquet, ORC) = read-heavy, analytics  
  - Row-based (JSON, CSV) = write-heavy, logs

- SageMaker Integrations:
  - Load data into SageMaker Data Wrangler  
  - Store features in SageMaker Feature Store

- Merging Data:
  - Use AWS Glue, Spark on EMR, or pandas

- Troubleshooting:
  - Capacity: check storage throughput limits (e.g., EBS)
  - Scalability: S3 scales automatically, EFS throughput grows with usage

- Storage Decisions:
  - S3: Default, cost-effective
  - EBS: Block storage, high performance
  - EFS/FSx: Shared file systems for ML training

**Rule of Thumb**  
Parquet for Analytics: Choose Parquet or ORC for ML workflows needing efficient analytics and compression.  
S3 for Raw Data: Use S3 as the default storage layer unless latency/performance requires otherwise.

---

### Task 1.2: Transform Data and Perform Feature Engineering

#### Knowledge Of:
- Data Cleaning Techniques:
  - Detecting outliers, imputing missing values, deduplication, combining data

- Feature Engineering:
  - Scaling, standardization, binning, log transforms, normalization

- Encoding:
  - One-hot encoding, label encoding, binary encoding, tokenization

- Tools:
  - SageMaker Data Wrangler
  - AWS Glue and Glue DataBrew (code/no-code)
  - AWS Lambda, Spark on EMR for stream transformation

- Labeling Services:
  - SageMaker Ground Truth
  - Amazon Mechanical Turk

#### Skills In:
- Transforming Data:
  - Glue for scalable ETL  
  - Data Wrangler for fast exploration  
  - Spark on EMR for big data

- Creating Features:
  - Use SageMaker Feature Store to register and reuse features

- Labeling:
  - Use SageMaker Ground Truth with built-in workflows

**Rule of Thumb**  
Data Wrangler for Rapid Prototyping: Use for visual data prep and export to SageMaker pipelines.  
Glue for Scalable ETL: Best for large-scale or scheduled data transformations.

---

### Task 1.3: Ensure Data Integrity and Prepare Data for Modeling

#### Knowledge Of:
- Bias Metrics:
  - Class Imbalance (CI)
  - Difference in Proportions of Labels (DPL)

- Bias Mitigation Strategies:
  - Resampling, Synthetic data generation
  - Use SageMaker Clarify for bias detection

- Data Security:
  - Encryption: S3 SSE, KMS
  - Masking, anonymization

- Compliance Considerations:
  - PII, PHI, data residency

#### Skills In:
- Data Quality Validation:
  - AWS Glue Data Quality
  - AWS Glue DataBrew

- Bias Detection and Mitigation:
  - Use SageMaker Clarify
  - Apply data augmentation, shuffling, class balancing

- Preparing for Training:
  - Use Amazon EFS or FSx as input sources
  - Ensure correct format and folder structure for training jobs

**Rule of Thumb**  
Clarify for Bias: Use SageMaker Clarify to measure and mitigate pre-training bias.  
Data Quality: Use Glue Data Quality or DataBrew for validating schema, completeness, and integrity.

---

## Domain 2: ML Model Development

---

### Task 2.1: Choose a Modeling Approach

#### Knowledge Of:
- ML Algorithm Capabilities:
  - Classification, regression, clustering, time series forecasting
  - Trade-offs: performance, interpretability, complexity, cost

- AWS AI Services:
  - Amazon Translate: Text translation
  - Amazon Transcribe: Speech-to-text
  - Amazon Rekognition: Image/video analysis
  - Amazon Bedrock: Foundation model (FM) access (Claude, Titan, etc.)

- Interpretability:
  - Consider explainability when choosing models (e.g., avoid deep nets for regulated environments)

- SageMaker Built-in Algorithms:
  - XGBoost, Linear Learner, KNN, Factorization Machines, BlazingText
  - Best for rapid prototyping with minimal code

#### Skills In:
- Assess Feasibility:
  - Use business constraints and data availability to assess model viability

- Choose Appropriate Models:
  - Classification → Logistic regression, XGBoost
  - Text Gen → Foundation Models in Bedrock
  - Image → CNNs or Rekognition

- Use Pre-built Tools:
  - JumpStart: Solution templates with pretrained models
  - Bedrock: Use if pre-trained FM can meet needs faster and cheaper

- Optimize for Cost:
  - Lightweight models for inference on edge devices
  - Prefer AI services over building custom models if the task is common

**Rule of Thumb**  
Use AI Services (Translate, Transcribe, Rekognition) when you don't need custom training.  
Use Bedrock or JumpStart to quickly prototype with Foundation Models or pre-trained templates.

---

### Task 2.2: Train and Refine Models

#### Knowledge Of:
- Training Concepts:
  - Epochs, batch size, steps affect training time and accuracy

- Reduce Training Time:
  - Use early stopping, distributed training, mixed precision

- Improve Model Performance:
  - Hyperparameter tuning, feature engineering, model selection
  - Regularization (dropout, L1/L2, weight decay) to avoid overfitting

- Hyperparameter Tuning:
  - SageMaker AMT supports random search, Bayesian optimization

- External Model Integration:
  - Import models via script mode or model.tar.gz for custom frameworks

#### Skills In:
- Use SageMaker Built-in Algorithms or ML Frameworks (TensorFlow, PyTorch)
- Fine-Tune Pre-trained Models using custom data (JumpStart, Bedrock)
- Apply Hyperparameter Tuning with SageMaker AMT
- Prevent Overfitting:
  - Use regularization, early stopping, dropout
- Model Ensembling:
  - Combine models (bagging, boosting, stacking) for better performance
- Reduce Model Size:
  - Quantization, pruning, compressing inputs/features
- Version Management:
  - Use SageMaker Model Registry to track versions and enable audits

**Rule of Thumb**  
Use SageMaker AMT for hands-off hyperparameter tuning.  
Reduce overfitting with dropout and regularization, underfitting with deeper models or more features.

---

### Task 2.3: Analyze Model Performance

#### Knowledge Of:
- Evaluation Metrics:
  - Classification: Accuracy, Precision, Recall, F1 Score, AUC-ROC
  - Regression: RMSE, MAE
  - Visualization: Confusion matrix, ROC curves, heat maps

- Baselines:
  - Use null models or historical data for baseline comparison

- Overfitting/Underfitting Detection:
  - High train/low test accuracy = overfit  
  - Low train accuracy = underfit

- Interpretability with Clarify:
  - SHAP values, bias metrics

- Convergence Issues:
  - Use SageMaker Debugger to identify stalled or unstable training

#### Skills In:
- Select Metrics Based on Problem Type:
  - Imbalanced classes → use F1, precision/recall
  - Continuous output → use RMSE, MAE

- Evaluate Trade-offs:
  - Training time vs performance vs cost

- Reproducibility:
  - Use SageMaker Experiments for tracking runs

- Shadow Testing:
  - Compare production and shadow variants (A/B testing)

- Debug Models:
  - Use SageMaker Debugger to inspect gradients, loss functions

**Rule of Thumb**  
Use F1 or AUC-ROC for imbalanced classification problems.  
Use SageMaker Clarify for model bias and interpretability; Debugger for training issues.

---

## Domain 3: Deployment and Orchestration of ML Workflows

---

### Task 3.1: Select Deployment Infrastructure Based on Existing Architecture and Requirements

#### Knowledge Of:
- Deployment Best Practices:
  - Versioning, rollback strategies, blue/green, canary deployments

- AWS Deployment Services:
  - SageMaker: Supports real-time, async, and batch inference endpoints

- Serving Models:
  - Real-time: Low latency, synchronous (use SageMaker real-time endpoints)
  - Async: For long-running jobs (SageMaker async endpoints)
  - Batch: For large batch scoring (SageMaker batch transform)

- Compute Provisioning:
  - Choose GPU for deep learning models, CPU for lightweight inference
  - Match instance family to model size and inference latency needs

- Endpoint Types:
  - Real-time endpoints
  - Asynchronous endpoints
  - Batch transform
  - Serverless endpoints for infrequent low-latency needs

- Containers:
  - Use SageMaker pre-built containers or bring your own (BYOC)

- Edge Optimization:
  - Use SageMaker Neo to compile and optimize models for edge deployment

#### Skills In:
- Evaluate Trade-offs:
  - Cost vs latency vs throughput
  - Spot vs On-Demand

- Choose Deployment Targets:
  - SageMaker endpoints, Lambda (for light models), ECS/EKS for containerized workflows

- Orchestration Tools:
  - SageMaker Pipelines: ML-specific CI/CD  
  - Apache Airflow: General workflow orchestration (via MWAA)

- Multi-model Hosting:
  - Serve multiple models behind a single endpoint

**Rule of Thumb**  
Use SageMaker real-time endpoints for low-latency inference, async for large/predictable delay tasks, batch for offline scoring.  
Use SageMaker Neo when deploying models to edge devices.

---

### Task 3.2: Create and Script Infrastructure Based on Existing Architecture and Requirements

#### Knowledge Of:
- Resource Types:
  - On-demand: Pay-per-use, no upfront commitment  
  - Provisioned: Reserved capacity with guaranteed availability

- Scaling Policies:
  - Automatic scaling (invocations per instance, CPU utilization, latency)
  - Manual and scheduled scaling

- Infrastructure as Code (IaC):
  - CloudFormation: Declarative templates  
  - AWS CDK: Imperative IaC with familiar languages

- Containerization:
  - Use ECR to store images  
  - ECS/EKS for container deployment  
  - BYOC to bring custom environments to SageMaker

- SageMaker Auto Scaling:
  - Use endpoint invocation metrics to auto-scale based on traffic or time

#### Skills In:
- Enable Cost-effective Scaling:
  - Use Spot Instances for training  
  - Use Lambda + SageMaker endpoint for low-frequency workloads

- Automate Provisioning:
  - CloudFormation/CDK to deploy entire ML infrastructure stacks

- Container Management:
  - Build containers, push to ECR, deploy with ECS/EKS or SageMaker

- VPC Configurations:
  - Configure SageMaker endpoints to run inside a VPC for security

- SageMaker SDK:
  - Use SDK to deploy, update, and manage model endpoints

**Rule of Thumb**  
Use auto-scaling for inference endpoints to balance cost and performance.  
Use CloudFormation/CDK for reproducible and automated infrastructure deployments.

---

### Task 3.3: Use Automated Orchestration Tools to Set Up CI/CD Pipelines

#### Knowledge Of:
- Code Tools:
  - CodePipeline: CI/CD orchestration  
  - CodeBuild: Build & test stages  
  - CodeDeploy: Deployment stages

- CI/CD in ML:
  - Includes model retraining, testing, validation, and deployment

- Deployment Strategies:
  - Canary, blue/green, and linear rollouts for safe updates

- Code Repositories:
  - Git/GitHub as version control and CI/CD triggers

#### Skills In:
- Pipeline Configuration:
  - Define stages for training, evaluation, deployment

- Git-Based Triggers:
  - Use GitHub Flow or Gitflow for CI/CD pipeline branching and release

- Automate Workflows:
  - Use SageMaker Pipelines or EventBridge to trigger retraining or inference

- Testing:
  - Unit, integration, and end-to-end tests in model deployment pipelines

- Retraining Triggers:
  - Build pipelines that automatically retrain models when data or code changes

**Rule of Thumb**  
Use SageMaker Pipelines for ML-specific CI/CD with built-in tracking.  
Use CodePipeline + CodeBuild + CodeDeploy for general purpose CI/CD and integration with Git.

---

## Domain 4: ML Solution Monitoring, Maintenance, and Security

---

### Task 4.1: Monitor Model Inference

#### Knowledge Of:
- Model Drift:
  - Concept drift: relationship between input and output changes
  - Data drift: input distribution changes over time

- Monitoring Techniques:
  - Track data quality, prediction distributions, feature skew

- ML Lens Principles:
  - Design for observability, detectability, and traceability in ML workflows

#### Skills In:
- Use SageMaker Model Monitor:
  - Monitor input data, prediction distributions, feature attribution drift

- Detect Anomalies:
  - Set alerts for data pipeline errors or missing features

- Monitor with Clarify:
  - Analyze post-deployment bias and data drift

- A/B Testing:
  - Deploy shadow models to monitor performance before full rollout

**Rule of Thumb**  
Use SageMaker Model Monitor for drift detection and prediction monitoring.  
Use Clarify for bias and drift analysis during and after deployment.

---

### Task 4.2: Monitor and Optimize Infrastructure and Costs

#### Knowledge Of:
- ML Infra Metrics:
  - Utilization, latency, throughput, fault tolerance

- Monitoring Tools:
  - AWS X-Ray: Trace app performance  
  - CloudWatch Logs & Insights: Log aggregation and queries  
  - Lambda Insights: Function-specific monitoring

- Logging and Audit:
  - Use AWS CloudTrail for tracking resource usage and security events

- Instance Types:
  - Memory optimized: R-series  
  - Compute optimized: C-series  
  - General purpose: T-series  
  - Inference optimized: Inf1, G5

- Cost Management Tools:
  - Cost Explorer, Budgets, Trusted Advisor, Billing & Cost Management

#### Skills In:
- Use CloudWatch Dashboards:
  - Visualize performance metrics

- Configure Logs and Alarms:
  - Trigger alerts based on latency, throughput, or cost spikes

- CloudTrail Integration:
  - Log activities and invoke retraining via EventBridge

- Instance Right-Sizing:
  - Use SageMaker Inference Recommender  
  - Use Compute Optimizer for instance recommendations

- Cost Optimization:
  - Spot for training, Reserved for predictable workloads  
  - Apply tagging strategy for allocation

**Rule of Thumb**  
Use CloudWatch for monitoring, CloudTrail for auditing, and Cost Explorer for financial analysis.  
Use Spot for training, and Reserved or Savings Plans for steady workloads.

---

### Task 4.3: Secure AWS Resources

#### Knowledge Of:
- IAM Roles and Policies:
  - Use least privilege principles  
  - Role-based access for SageMaker, S3, ECR, etc.

- SageMaker Security Features:
  - SageMaker Role Manager for access setup  
  - Model encryption with KMS  
  - Endpoint VPC isolation

- Network Controls:
  - Use VPC, subnets, security groups to isolate ML workloads

- Secure CI/CD:
  - Secure CodePipeline stages with IAM, encryption, and audits

#### Skills In:
- Configure IAM:
  - Role assumption, trust relationships, policy scoping

- Audit and Monitor:
  - Use CloudTrail for compliance  
  - Set up CloudWatch and EventBridge for monitoring anomalies

- Secure Networking:
  - Deploy endpoints within VPC  
  - Restrict access with security groups and NACLs

- Debug Security:
  - Trace IAM permission errors and troubleshoot network isolation issues

**Rule of Thumb**  
Always use least privilege IAM policies and isolate endpoints in VPCs.  
Enable encryption and logging to maintain compliance and traceability.

---

## Appendix: In-Scope AWS Services for MLA-C01

---

### Analytics

- **Amazon Athena**: Serverless SQL over S3 data.
- **Amazon Data Firehose**: Stream data into S3/Redshift with optional transform.
- **Amazon EMR**: Managed big data processing (Spark, Hadoop).
- **AWS Glue**: Serverless ETL for large-scale data prep.
- **AWS Glue DataBrew**: No-code data wrangling.
- **AWS Glue Data Quality**: Rule-based data validation and profiling.
- **Amazon Kinesis**: Real-time data streaming.
- **AWS Lake Formation**: Secure and govern S3 data lakes.
- **Amazon Managed Service for Apache Flink**: Real-time stream processing.
- **Amazon OpenSearch Service**: Search and analytics engine.
- **Amazon QuickSight**: BI and dashboards over AWS data.
- **Amazon Redshift**: Scalable data warehouse with ML integration.

---

### Application Integration

- **Amazon EventBridge**: Event bus for triggering workflows.
- **Amazon MWAA**: Managed Apache Airflow for orchestration.
- **Amazon SNS**: Pub/sub messaging for decoupled systems.
- **Amazon SQS**: Message queue for decoupling components.
- **AWS Step Functions**: Visual workflow orchestration.

---

### Cloud Financial Management

- **AWS Billing and Cost Management**: View and manage charges.
- **AWS Budgets**: Set cost and usage alerts.
- **AWS Cost Explorer**: Visualize usage and spending trends.

---

### Compute

- **AWS Batch**: Run batch jobs on managed compute.
- **Amazon EC2**: Scalable virtual machines.
- **AWS Lambda**: Serverless compute for event-driven ML.
- **AWS Serverless Application Repository**: Deploy Lambda-based blueprints.

---

### Containers

- **Amazon ECR**: Container image registry.
- **Amazon ECS**: Managed container orchestration.
- **Amazon EKS**: Managed Kubernetes service.

---

### Database

- **Amazon DocumentDB**: Managed MongoDB-compatible DB.
- **Amazon DynamoDB**: Serverless key-value and document DB.
- **Amazon ElastiCache**: In-memory caching (Redis/Memcached).
- **Amazon Neptune**: Graph database.
- **Amazon RDS**: Managed relational databases (MySQL, Postgres, etc.).

---

### Developer Tools

- **AWS CDK**: Code-based IaC using TypeScript, Python, etc.
- **AWS CodeArtifact**: Package management for code dependencies.
- **AWS CodeBuild**: Build and test automation.
- **AWS CodeDeploy**: Automate deployment to compute targets.
- **AWS CodePipeline**: CI/CD pipeline service.
- **AWS X-Ray**: Trace application and ML pipeline latency.

---

### Machine Learning

- **Amazon A2I**: Human-in-the-loop review for ML predictions.
- **Amazon Bedrock**: Access to foundation models via API.
- **Amazon CodeGuru**: ML-based code reviews and profiling.
- **Amazon Comprehend**: NLP for sentiment, entity detection.
- **Amazon Comprehend Medical**: NLP for healthcare data.
- **Amazon DevOps Guru**: Operational ML anomaly detection.
- **Amazon Fraud Detector**: Build fraud detection models.
- **AWS HealthLake**: Store and analyze healthcare data (FHIR).
- **Amazon Kendra**: Intelligent search engine.
- **Amazon Lex**: Conversational chatbot engine.
- **Amazon Lookout for Equipment**: Predictive maintenance for IoT.
- **Amazon Lookout for Metrics**: Anomaly detection in metrics.
- **Amazon Lookout for Vision**: Visual anomaly detection.
- **Amazon Mechanical Turk**: Human labeling workforce.
- **Amazon Personalize**: Recommendation engine.
- **Amazon Polly**: Text-to-speech.
- **Amazon Q**: Generative AI-powered assistant.
- **Amazon Rekognition**: Image/video analysis.
- **Amazon SageMaker**: Full ML lifecycle platform.
- **Amazon Textract**: Extract text/structured data from docs.
- **Amazon Transcribe**: Speech-to-text.
- **Amazon Translate**: Language translation.

---

### Management and Governance

- **AWS Auto Scaling**: Scale resources dynamically.
- **AWS Chatbot**: Alerts and notifications in Slack/Chime.
- **AWS CloudFormation**: Declarative infrastructure as code.
- **AWS CloudTrail**: Audit logs for actions on AWS.
- **Amazon CloudWatch**: Metrics, logging, dashboards.
- **Amazon CloudWatch Logs**: Store and analyze logs.
- **AWS Compute Optimizer**: Instance size/cost recommendations.
- **AWS Config**: Track configuration changes.
- **AWS Organizations**: Manage multiple AWS accounts.
- **AWS Service Catalog**: Pre-approved infrastructure templates.
- **AWS Systems Manager**: Operational insights and automation.
- **AWS Trusted Advisor**: Best practice and cost checks.

---

### Media

- **Amazon Kinesis Video Streams**: Ingest and process video streams.

---

### Migration and Transfer

- **AWS DataSync**: Accelerate on-prem to cloud data transfers.

---

### Networking and Content Delivery

- **Amazon API Gateway**: Create and manage APIs.
- **Amazon CloudFront**: CDN for content acceleration.
- **AWS Direct Connect**: Private network link to AWS.
- **Amazon VPC**: Isolated network for resources.

---

### Security, Identity, and Compliance

- **AWS IAM**: Identity and access management.
- **AWS KMS**: Encryption key management.
- **Amazon Macie**: Managed PII detection in S3.
- **AWS Secrets Manager**: Store and rotate credentials.

---

### Storage

- **Amazon EBS**: Block storage for EC2.
- **Amazon EFS**: Scalable shared file storage.
- **Amazon FSx**: Managed Windows/NetApp file systems.
- **Amazon S3**: Scalable object storage.
- **Amazon S3 Glacier**: Archival storage.
- **AWS Storage Gateway**: Hybrid storage integration.

---
