using System.Diagnostics;
using UdcFileEncryption.Models;
using UdcFileEncryption.Services;

// ============================================================
// UDC File Encryption — C:\AI\FILES
// ============================================================

//const string publicKeyPath  = @"C:\Repos\UDC\UDC-KEY-120.pem";
const string publicKeyPath  = @"C:\Repos\UDC\V1-KEY.pem";
const string inputDir       = @"C:\AI\FILES";
const string siteId         = "SITE-1";
const string keyVersion     = "V1-KEY";

// Full Key Vault key identifier — include the version segment so KV uses the exact key
// Format: https://<vault>.vault.azure.net/keys/<key-name>/<version>
const string keyVaultKeyId  = "https://ay-demo-kv.vault.azure.net/keys/V1-KEY/fa9a4ee258534e0d992a055db9db604d";

if (!Directory.Exists(inputDir))
{
    Console.WriteLine($"ERROR: Input directory not found at {inputDir}");
    return;
}

Console.WriteLine("╔══════════════════════════════════════╗");
Console.WriteLine("║   UDC File Encryption Tool           ║");
Console.WriteLine("╠══════════════════════════════════════╣");
Console.WriteLine("║  E  —  Encrypt files                 ║");
Console.WriteLine("║  D  —  Decrypt files                 ║");
Console.WriteLine("╚══════════════════════════════════════╝");
Console.Write("\nChoice: ");
string? choice = Console.ReadLine()?.Trim().ToUpperInvariant();
Console.WriteLine();

if (choice == "E")
{
    // ---- ENCRYPT ------------------------------------------------
    if (!File.Exists(publicKeyPath))
    {
        Console.WriteLine($"ERROR: Public key not found at {publicKeyPath}");
        return;
    }

    // Skip files that are already encrypted (stem ends with _enc)
    var filesToEncrypt = Directory.GetFiles(inputDir)
        .Where(f => !Path.GetFileNameWithoutExtension(f).EndsWith("_enc", StringComparison.OrdinalIgnoreCase))
        .ToArray();

    if (filesToEncrypt.Length == 0)
    {
        Console.WriteLine($"No files to encrypt in {inputDir}");
        return;
    }

    Console.WriteLine($"Found {filesToEncrypt.Length} file(s) to encrypt\n");

    var config = new UdcEncryptionConfig
    {
        SiteId         = siteId,
        KeyVersion     = keyVersion,
        PublicKeyPemPath = publicKeyPath,
        KeyExpiresAt   = DateTime.UtcNow.AddYears(1)
    };
    var encryptor = new OnPremEncryptionService(config);

    foreach (var filePath in filesToEncrypt)
    {
        var fileInfo = new FileInfo(filePath);
        string nameWithoutExt = Path.GetFileNameWithoutExtension(fileInfo.Name);
        string outputPath = Path.Combine(inputDir, $"{nameWithoutExt}_enc{fileInfo.Extension}");

        Console.WriteLine($"Encrypting: {fileInfo.Name} ({fileInfo.Length:N0} bytes)");

        var sw = Stopwatch.StartNew();
        encryptor.EncryptFile(filePath, outputPath, "UDC_FILE");
        sw.Stop();

        Console.WriteLine($"  -> {Path.GetFileName(outputPath)} ({new FileInfo(outputPath).Length:N0} bytes) [{sw.ElapsedMilliseconds}ms]\n");
    }
}
else if (choice == "D")
{
    // ---- DECRYPT via Azure Key Vault ------------------------------------------------
    if (keyVaultKeyId.Contains("<your-vault>"))
    {
        Console.WriteLine("ERROR: Set keyVaultKeyId in Program.cs before decrypting.");
        Console.WriteLine("  Format: https://<vault>.vault.azure.net/keys/<key-name>/<version>");
        return;
    }

    var filesToDecrypt = Directory.GetFiles(inputDir)
        .Where(f => Path.GetFileNameWithoutExtension(f).EndsWith("_enc", StringComparison.OrdinalIgnoreCase))
        .ToArray();

    if (filesToDecrypt.Length == 0)
    {
        Console.WriteLine($"No *_enc.* files found in {inputDir}");
        return;
    }

    Console.WriteLine($"Found {filesToDecrypt.Length} encrypted file(s)\n");

    var decryptor = new KeyVaultDecryptionService(keyVaultKeyId);

    foreach (var filePath in filesToDecrypt)
    {
        var fileInfo = new FileInfo(filePath);
        // e.g. report_enc.csv -> report_dec.csv
        string stem = Path.GetFileNameWithoutExtension(fileInfo.Name); // "report_enc"
        string outputName = stem[..^4] + "_dec" + fileInfo.Extension;  // chop "_enc", add "_dec"
        string outputPath = Path.Combine(inputDir, outputName);

        Console.WriteLine($"Decrypting: {fileInfo.Name} ({fileInfo.Length:N0} bytes)");

        try
        {
            var sw = Stopwatch.StartNew();
            byte[] plaintext = await decryptor.DecryptFileAsync(filePath, siteId);
            sw.Stop();

            File.WriteAllBytes(outputPath, plaintext);
            Console.WriteLine($"  -> {outputName} ({plaintext.Length:N0} bytes) [{sw.ElapsedMilliseconds}ms]\n");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  ERROR: {ex.GetType().Name}: {ex.Message}\n");
        }
    }
}
else
{
    Console.WriteLine("Invalid choice. Please enter E or D.");
    return;
}

Console.WriteLine("Done.");
