using Azure.Identity;
using Azure.Security.KeyVault.Keys.Cryptography;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using UdcFileEncryption.Models;

namespace UdcFileEncryption.Services;

/// <summary>
/// Decrypts UDC files where the RSA private key lives in Azure Key Vault (HSM).
/// The encrypted AES envelope is sent to KV's Decrypt API — the private key never leaves the HSM.
/// </summary>
public class KeyVaultDecryptionService
{
    private readonly CryptographyClient _cryptoClient;

    /// <summary>
    /// keyVaultKeyId — full Key Vault key identifier, e.g.
    ///   https://my-kv.vault.azure.net/keys/udc-file-encryption-key/&lt;version&gt;
    /// Authentication uses DefaultAzureCredential (VS login / az login / Managed Identity).
    /// </summary>
    public KeyVaultDecryptionService(string keyVaultKeyId)
    {
        _cryptoClient = new CryptographyClient(new Uri(keyVaultKeyId), new DefaultAzureCredential());
        Console.WriteLine($"[KV] CryptographyClient ready for key: {keyVaultKeyId}");
    }

    /// <summary>
    /// Decrypts a UDC encrypted file using Azure Key Vault for the RSA unwrap step.
    ///
    /// Steps:
    ///   1. Read plaintext header — validate SiteId
    ///   2. Send encrypted envelope to KV Decrypt API — receive AES key + nonce
    ///   3. Decrypt content locally with AES-256-GCM — header AAD verified
    /// </summary>
    public async Task<byte[]> DecryptFileAsync(string encryptedFilePath, string expectedSiteId)
    {
        byte[] fileBytes = File.ReadAllBytes(encryptedFilePath);
        Console.WriteLine($"[KV] Reading encrypted file: {encryptedFilePath} ({fileBytes.Length:N0} bytes)");

        // === Step 1: Read plaintext header ===
        int newlinePos = Array.IndexOf(fileBytes, (byte)'\n');
        if (newlinePos < 0)
            throw new InvalidDataException("Invalid UDC encrypted file: no header found");

        byte[] headerBytes = fileBytes[..newlinePos];
        string headerJson = Encoding.UTF8.GetString(headerBytes);
        var header = JsonSerializer.Deserialize<EncryptedFileHeader>(headerJson)
            ?? throw new InvalidDataException("Invalid header JSON");

        Console.WriteLine($"[KV] Header: version={header.Version}, site={header.SiteId}, " +
                          $"ts={header.Timestamp}, cat={header.Category}");

        // Validate site BEFORE any KV call
        if (!string.Equals(header.SiteId, expectedSiteId, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException(
                $"WRONG SITE: File belongs to '{header.SiteId}' but expected '{expectedSiteId}'. Upload rejected.");
        }
        Console.WriteLine($"[KV] Site ID validated: {header.SiteId} ✓");

        int offset = newlinePos + 1;

        // === Step 2: Read encrypted envelope ===
        uint envelopeLength = BitConverter.ToUInt32(fileBytes, offset);
        offset += 4;

        byte[] encryptedEnvelope = fileBytes[offset..(offset + (int)envelopeLength)];
        offset += (int)envelopeLength;

        // Send to Key Vault — private key never leaves HSM
        Console.WriteLine("[KV] Sending envelope to Key Vault for RSA-OAEP-SHA256 decryption...");
        var decryptResult = await _cryptoClient.DecryptAsync(
            EncryptionAlgorithm.RsaOaep256,
            encryptedEnvelope);

        byte[] keyMaterial = decryptResult.Plaintext;
        byte[] aesKey = keyMaterial[..32];
        byte[] nonce  = keyMaterial[32..44];
        Console.WriteLine($"[KV] Envelope decrypted by KV: AES key ({aesKey.Length * 8}-bit) + nonce ({nonce.Length} bytes) ✓");

        // === Step 3: Read auth tag + ciphertext ===
        byte[] authTag    = fileBytes[offset..(offset + 16)];
        offset += 16;
        byte[] ciphertext = fileBytes[offset..];

        // Decrypt locally with AES-256-GCM
        byte[] plaintext = new byte[ciphertext.Length];
        using (var aesGcm = new AesGcm(aesKey, 16))
        {
            aesGcm.Decrypt(
                nonce: nonce,
                ciphertext: ciphertext,
                tag: authTag,
                plaintext: plaintext,
                associatedData: headerBytes);
        }

        CryptographicOperations.ZeroMemory(aesKey);
        CryptographicOperations.ZeroMemory(keyMaterial);

        Console.WriteLine($"[KV] Decrypted successfully: {plaintext.Length:N0} bytes ✓");
        return plaintext;
    }
}
