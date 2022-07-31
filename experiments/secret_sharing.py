from secretsharing import PlaintextToHexSecretSharer


def split_data_into_secret(data, threshold, secret_num):
    # secrets = PlaintextToHexSecretSharer.split_secret(data, threshold, secret_num)
    secrets = [[] for _ in range(secret_num)]
    for i in data:
        shares = PlaintextToHexSecretSharer.split_secret(str(i), threshold, secret_num)
        for j in range(secret_num):
            secrets[j].append(shares[j])
    return secrets


def recover_secret(secret_list):
    data = PlaintextToHexSecretSharer.recover_secret(secret_list)
    return data


if __name__ == '__main__':
    # shares = PlaintextToHexSecretSharer.split_secret("123.4", 2, 3)
    # print(shares)
    # print(PlaintextToHexSecretSharer.recover_secret(shares[0:2]))
    print(split_data_into_secret([155, 255, 553], 2, 3))