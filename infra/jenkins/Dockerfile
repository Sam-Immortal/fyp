# syntax=docker/dockerfile:1.7
FROM jenkins/jenkins:2.470-jdk17

USER root

# Install base tools required by CI/CD pipeline.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        gnupg \
        lsb-release \
        python3 \
        python3-pip \
        git \
    && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian bookworm stable" \
        > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker-ce-cli docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry for backend builds.
RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.8.2 \
    && ln -s /root/.local/bin/poetry /usr/local/bin/poetry || true

USER jenkins

COPY plugins.txt /usr/share/jenkins/ref/plugins.txt
RUN jenkins-plugin-cli --plugin-file /usr/share/jenkins/ref/plugins.txt

# Skip initial setup wizard so Jenkins comes up ready for JCasC.
ENV JAVA_OPTS="-Djenkins.install.runSetupWizard=false"
