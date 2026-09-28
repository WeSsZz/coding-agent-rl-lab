ARG BASE_IMAGE=ghcr.io/all-hands-ai/runtime:oh_v0.13.1_bn1t3fq68pl10p2f_y766p9z6jyxsud5c
FROM ${BASE_IMAGE}

COPY --from=ghcr.io/all-hands-ai/runtime:oh_v0.13.1_bn1t3fq68pl10p2f_y766p9z6jyxsud5c \
    /root/.cache/ms-playwright/ /root/.cache/ms-playwright/
