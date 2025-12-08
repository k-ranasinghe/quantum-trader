pipeline {
    agent any

    environment {
        DOCKER_IMAGE_NAME = "quantum-trader"
        REGISTRY_URL = "docker.io/yourusername" // Replace with actual registry
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'pip install pytest flake8'
            }
        }

        stage('Linting') {
            steps {
                // Continue on failure for now, just warn
                sh 'flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics || true'
                sh 'flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics'
            }
        }

        stage('Run Tests') {
            steps {
                sh 'pytest tests/ -v'
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("${DOCKER_IMAGE_NAME}:${BUILD_NUMBER}")
                }
            }
        }
    }

    post {
        always {
            cleanWs()
        }
        success {
            echo 'Pipeline succeeded!'
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
}
