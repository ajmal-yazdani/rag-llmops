using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using UdcFileEncryption.Models;

namespace UdcFileEncryption.Services;

/// <summary>
/// CLOUD-SIDE decryption service.
/// In production: uses Azure Key Vault's Decrypt API (private key never leaves HSM).
/// In this prototype: uses a local RSA private key for testing.
/// </summary>
public class CloudDecryptionService
{
    private readonly RSA _privateKey;

    /// <summary>
    /// Production constructor — would use Azure Key Vault client.
    /// </summary>
    public CloudDecryptionService(RSA privateKey)
    {
        _privateKey = privateKey;
        Console.WriteLine($"[Cloud] Loaded RSA private key: {_privateKey.KeySize}-bit");
    }

    /// <summary>
    /// Decrypts a UDC encrypted file.
    /// 
    /// Steps:
    ///   1. Read plaintext header — validate SiteId, get key version
    ///   2. Decrypt envelope with RSA private key — recover AES key + nonce
    ///   3. Decrypt content with AES-256-GCM — verify header integrity via AAD
    /// </summary>
    public byte[] DecryptFile(string encryptedFilePath, string expectedSiteId)
    {
        byte[] fileBytes = File.ReadAllBytes(encryptedFilePath);
        Console.WriteLine($"[Cloud] Reading encrypted file: {encryptedFilePath} ({fileBytes.Length:N0} bytes)");

        int offset = 0;

        // === Step 1: Read plaintext header (everything before first \n) ===
        int newlinePos = Array.IndexOf(fileBytes, (byte)'\n');
        if (newlinePos < 0)
            throw new InvalidDataException("Invalid UDC encrypted file: no header found");

        byte[] headerBytes = fileBytes[..newlinePos];
        string headerJson = Encoding.UTF8.GetString(headerBytes);
        var header = JsonSerializer.Deserialize<EncryptedFileHeader>(headerJson)
            ?? throw new InvalidDataException("Invalid header JSON");

        Console.WriteLine($"[Cloud] Header: version={header.Version}, site={header.SiteId}, " +
                          $"ts={header.Timestamp}, cat={header.Category}");

        // === Step 1b: Validate Site ID (BEFORE any decryption / Key Vault calls) ===
        if (!string.Equals(header.SiteId, expectedSiteId, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException(
                $"WRONG SITE: File belongs to '{header.SiteId}' but expected '{expectedSiteId}'. " +
                "Upload rejected.");
        }
        Console.WriteLine($"[Cloud] Site ID validated: {header.SiteId} ✓");

        offset = newlinePos + 1;

        // === Step 2: Read envelope ===
        uint envelopeLength = BitConverter.ToUInt32(fileBytes, offset);
        offset += 4;

        byte[] encryptedEnvelope = fileBytes[offset..(offset + (int)envelopeLength)];
        offset += (int)envelopeLength;

        // Decrypt envelope with RSA private key
        // PRODUCTION: byte[] keyMaterial = await keyVaultClient.DecryptAsync(keyName, encryptedEnvelope);
        byte[] keyMaterial = _privateKey.Decrypt(encryptedEnvelope, RSAEncryptionPadding.OaepSHA256);

        byte[] aesKey = keyMaterial[..32];
        byte[] nonce = keyMaterial[32..44];

        Console.WriteLine($"[Cloud] Envelope decrypted: AES key ({aesKey.Length * 8}-bit) + nonce ({nonce.Length} bytes) ✓");

        // === Step 3: Read auth tag + ciphertext ===
        byte[] authTag = fileBytes[offset..(offset + 16)];
        offset += 16;

        byte[] ciphertext = fileBytes[offset..];

        // Decrypt with AES-256-GCM, using header as AAD for tamper detection
        byte[] plaintext = new byte[ciphertext.Length];
        using (var aesGcm = new AesGcm(aesKey, 16))
        {
            aesGcm.Decrypt(
                nonce: nonce,
                ciphertext: ciphertext,
                tag: authTag,
                plaintext: plaintext,
                associatedData: headerBytes  // If header was tampered, this throws
            );
        }

        // Clear sensitive material
        CryptographicOperations.ZeroMemory(aesKey);
        CryptographicOperations.ZeroMemory(keyMaterial);

        Console.WriteLine($"[Cloud] Decrypted successfully: {plaintext.Length:N0} bytes ✓");
        return plaintext;
    }
}
