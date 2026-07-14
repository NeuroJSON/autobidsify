# universal_core.py
# Universal file analysis and matching engine - zero assumptions, data-driven

"""
Universal File Analysis and Matching Engine

Design Philosophy:
- Python handles precise data analysis and statistics
- LLM handles semantic understanding and decision making
- Matching uses structured features, not fragile regex

Core Classes:
1. FileStructureAnalyzer - Analyze file structure, detect subjects
2. UniversalFileMatcher - Zero-regex feature-based matching
3. SmartFileGrouper - Intelligent grouping and deduplication
"""

from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict, Counter
import re


class FileStructureAnalyzer:
    """
    File structure analyzer with zero assumptions.
    
    Capabilities:
    1. Analyze directory hierarchy
    2. Detect subject identifiers automatically
    3. Identify file naming patterns
    4. Detect duplicate filenames across paths
    """
    
    def __init__(self, all_file_paths: List[str]):
        """
        Args:
            all_file_paths: List of all relative file paths (as strings)
        """
        self.all_files = all_file_paths
        self.analysis_cache = {}
    
    def analyze_directory_structure(self) -> Dict[str, Any]:
        """
        Analyze directory hierarchy structure.
        
        Returns:
            {
                "max_depth": int,
                "depth_distribution": {depth: count},
                "unique_dir_names": List[str],
                "dir_level_patterns": {level: [names]},
                "structure_template": str
            }
        """
        if "dir_structure" in self.analysis_cache:
            return self.analysis_cache["dir_structure"]
        
        depth_counter = Counter()
        unique_dirs = set()
        level_dirs = defaultdict(set)
        
        for filepath in self.all_files:
            parts = filepath.split('/')
            depth = len(parts) - 1  # Exclude filename
            depth_counter[depth] += 1
            
            # Collect directory names at each level
            for level, part in enumerate(parts[:-1]):
                unique_dirs.add(part)
                level_dirs[level].add(part)
        
        # Infer structure template
        template = self._infer_structure_template(level_dirs)
        
        result = {
            "max_depth": max(depth_counter.keys()) if depth_counter else 0,
            "depth_distribution": dict(depth_counter),
            "unique_dir_names": sorted(unique_dirs)[:100],
            "dir_level_patterns": {k: sorted(v)[:20] for k, v in level_dirs.items()},
            "total_unique_dirs": len(unique_dirs),
            "structure_template": template
        }
        
        self.analysis_cache["dir_structure"] = result
        return result
    
    def _infer_structure_template(self, level_dirs: Dict[int, Set[str]]) -> str:
        """Infer directory structure template from patterns"""
        if not level_dirs or 0 not in level_dirs:
            return "flat"
        
        # Check first level directory names
        first_level_sample = list(level_dirs[0])[:10]
        
        has_sub_keyword = any('sub' in d.lower() for d in first_level_sample)
        
        if has_sub_keyword:
            if len(level_dirs) == 1:
                return "{subject}"
            elif len(level_dirs) == 2:
                return "{subject}/{scantype}"
            elif len(level_dirs) == 3:
                return "{subject}/{scantype}/{format}"
            else:
                return "{subject}/nested"
        else:
            return f"custom_{len(level_dirs)}_levels"
    
    def detect_subject_identifiers(self, user_hint: Optional[int] = None) -> Dict[str, Any]:
        """
        Detect all possible subject identifier patterns - zero assumptions.
        
        Strategy:
        1. Try common directory patterns (Site_subID, sub-ID, numeric)
        2. Try filename patterns (patient_ID, subject_ID)
        3. Score each candidate based on multiple criteria
        4. Return best candidate with extraction regex (already escape-fixed!)
        
        Args:
            user_hint: User-provided subject count hint
        
        Returns:
            {
                "candidates": List[Dict],
                "best_candidate": Dict or None,
                "confidence": "high|medium|low|none"
            }
        """
        candidates = []
        
        # Extract patterns from directories
        dir_candidates = self._extract_directory_id_patterns()
        candidates.extend(dir_candidates)
        
        # Extract patterns from filenames
        filename_candidates = self._extract_filename_id_patterns()
        candidates.extend(filename_candidates)
        
        # Score all candidates
        for candidate in candidates:
            score = self._score_identifier_candidate(candidate, user_hint)
            candidate["score"] = score
        
        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Determine best and confidence
        best = candidates[0] if candidates else None
        
        if best and best["score"] > 80:
            confidence = "high"
        elif best and best["score"] > 60:
            confidence = "medium"
        elif best:
            confidence = "low"
        else:
            confidence = "none"
        
        return {
            "candidates": candidates[:5],
            "best_candidate": best,
            "confidence": confidence,
            "total_candidates_evaluated": len(candidates)
        }
    
    def _extract_directory_id_patterns(self) -> List[Dict]:
        """Extract subject ID patterns from directory names"""
        
        # Collect first-level directory names
        first_level_dirs = set()
        for filepath in self.all_files:
            parts = filepath.split('/')
            if len(parts) > 1:
                first_level_dirs.add(parts[0])
        
        if not first_level_dirs:
            return []
        
        candidates = []
        
        # Pattern 1: Site_subID (e.g., Beijing_sub82352)
        pattern1_matches = {}
        for dirname in first_level_dirs:
            match = re.match(r'([A-Za-z]+)_sub(\d+)', dirname, re.IGNORECASE)
            if match:
                site = match.group(1)
                subject_id = match.group(2)
                pattern1_matches[subject_id] = {"site": site, "original": dirname}
        
        if pattern1_matches:
            candidates.append({
                "type": "directory_pattern",
                "pattern_name": "site_sub_id",
                "pattern_display": "{site}_sub{id}",
                "extraction_regex": r'([A-Za-z]+)_sub(\d+)',  # Single backslash - correct for Python!
                "subject_group": 2,
                "site_group": 1,
                "count": len(pattern1_matches),
                "sample_ids": sorted(list(pattern1_matches.keys()))[:10],
                "metadata": {"has_site": True},
                "avg_files_per_subject": len(self.all_files) / len(pattern1_matches)
            })
        
        # Pattern 2: sub-ID or subID (BIDS standard)
        pattern2_matches = set()
        for dirname in first_level_dirs:
            match = re.match(r'sub-?(\d+)', dirname, re.IGNORECASE)
            if match:
                pattern2_matches.add(match.group(1))
        
        if pattern2_matches:
            candidates.append({
                "type": "directory_pattern",
                "pattern_name": "bids_standard",
                "pattern_display": "sub-{id}",
                "extraction_regex": r'sub-?(\d+)',
                "subject_group": 1,
                "site_group": None,
                "count": len(pattern2_matches),
                "sample_ids": sorted(list(pattern2_matches))[:10],
                "metadata": {"has_site": False},
                "avg_files_per_subject": len(self.all_files) / len(pattern2_matches)
            })
        
        # Pattern 3: Numeric directories (e.g., 001, 025, 12345)
        pattern3_matches = set()
        for dirname in first_level_dirs:
            if re.match(r'^\d{2,6}$', dirname):
                pattern3_matches.add(dirname)
        
        if pattern3_matches:
            candidates.append({
                "type": "directory_pattern",
                "pattern_name": "numeric_only",
                "pattern_display": "{id}",
                "extraction_regex": r'^(\d+)$',
                "subject_group": 1,
                "site_group": None,
                "count": len(pattern3_matches),
                "sample_ids": sorted(list(pattern3_matches))[:10],
                "metadata": {"numeric_only": True},
                "avg_files_per_subject": len(self.all_files) / len(pattern3_matches)
            })
        
        return candidates
    
    def _extract_filename_id_patterns(self) -> List[Dict]:
        """Extract subject ID patterns from filenames"""
        
        filenames = [fp.split('/')[-1] for fp in self.all_files]
        candidates = []
        
        # Pattern: patient_ID or subject_ID in filename
        pattern_matches = set()
        for filename in filenames:
            match = re.search(r'(?:patient|subject)[_-]?(\d+)', filename, re.IGNORECASE)
            if match:
                pattern_matches.add(match.group(1))
        
        if pattern_matches:
            candidates.append({
                "type": "filename_pattern",
                "pattern_name": "patient_or_subject_id",
                "pattern_display": "{prefix}_{id}",
                "extraction_regex": r'(?:patient|subject)[_-]?(\d+)',
                "subject_group": 1,
                "site_group": None,
                "count": len(pattern_matches),
                "sample_ids": sorted(list(pattern_matches))[:10],
                "metadata": {},
                "avg_files_per_subject": len(self.all_files) / len(pattern_matches)
            })
        
        return candidates
    
    def _score_identifier_candidate(self, candidate: Dict, user_hint: Optional[int]) -> float:
        """
        Score identifier candidate based on multiple criteria.
        
        Scoring dimensions:
        1. Count reasonableness (0-30 points)
        2. Consistency with user hint (0-30 points)
        3. File distribution uniformity (0-20 points)
        4. Pattern reliability (0-20 points)
        
        Returns:
            Score (0-100)
        """
        score = 0.0
        count = candidate["count"]
        avg_files = candidate.get("avg_files_per_subject", 0)
        
        # 1. Count reasonableness
        if 2 <= count <= 10:
            score += 25
        elif 10 < count <= 100:
            score += 30
        elif 100 < count <= 1000:
            score += 28
        elif 1000 < count <= 10000:
            score += 25
        else:
            score += 15
        
        # 2. Consistency with user hint
        if user_hint:
            if count == user_hint:
                score += 30
            else:
                ratio = min(count, user_hint) / max(count, user_hint)
                score += ratio * 25
        else:
            score += 15
        
        # 3. File distribution uniformity
        if 2 <= avg_files <= 10:
            score += 20
        elif avg_files > 10:
            score += 15
        else:
            score += 5
        
        # 4. Pattern type preference
        if candidate["type"] == "directory_pattern":
            score += 15
            if candidate.get("metadata", {}).get("has_site"):
                score += 5
        elif candidate["type"] == "filename_pattern":
            score += 10
        
        return round(score, 2)
    
    def detect_duplicate_filenames(self) -> Dict[str, List[str]]:
        """
        Detect duplicate filenames across different paths.
        
        Returns:
            {filename: [path1, path2, ...]} only for duplicates
        """
        filename_to_paths = defaultdict(list)
        
        for filepath in self.all_files:
            filename = filepath.split('/')[-1]
            filename_to_paths[filename].append(filepath)
        
        # Return only duplicates
        duplicates = {k: v for k, v in filename_to_paths.items() if len(v) > 1}
        
        return duplicates
    
    def build_directory_tree_summary(self, max_subjects: int = 50) -> Dict[str, Any]:
        """
        Build compressed directory tree summary for LLM.
        
        Strategy:
        1. Group by subject directory
        2. Sample representative subjects
        3. Store only filename patterns, not full paths
        
        This drastically reduces payload size for large datasets.
        
        Returns:
            Compressed structural representation
        """
        # Group by first-level directory (assumed to be subject)
        subject_to_structure = defaultdict(lambda: defaultdict(set))
        
        for filepath in self.all_files:
            parts = filepath.split('/')
            
            if len(parts) < 2:
                continue
            
            subject_dir = parts[0]
            remaining_path = '/'.join(parts[1:-1]) if len(parts) > 2 else ""
            filename = parts[-1]
            
            # Extract filename pattern (remove numbers, parentheses)
            filename_pattern = re.sub(r'\d+', 'N', filename)
            filename_pattern = re.sub(r'\s*\([^)]*\)', '', filename_pattern)
            
            subject_to_structure[subject_dir][remaining_path].add(filename_pattern)
        
        # Sample subjects
        all_subjects = sorted(subject_to_structure.keys())
        
        if len(all_subjects) <= max_subjects:
            sampled_subjects = all_subjects
        else:
            # Sample: beginning, middle, end
            sampled = []
            sampled.extend(all_subjects[:15])
            mid = len(all_subjects) // 2
            sampled.extend(all_subjects[mid-10:mid+10])
            sampled.extend(all_subjects[-15:])
            sampled_subjects = sorted(set(sampled))[:max_subjects]
        
        # Build summary
        summary = {}
        for subject in sampled_subjects:
            structure = {}
            for path, filenames in subject_to_structure[subject].items():
                path_key = path if path else "root"
                structure[path_key] = sorted(filenames)[:5]
            summary[subject] = structure
        
        return {
            "subject_structure_samples": summary,
            "total_subjects_detected": len(all_subjects),
            "sampled_subjects": len(sampled_subjects)
        }


class UniversalFileMatcher:
    """
    Universal file matcher using structured features instead of fragile regex.
    
    Core concept:
    - Patterns are semantic feature descriptions
    - Matching is based on keyword containment, not exact string matching
    - No regex parsing errors, no YAML escape issues
    """
    
    @staticmethod
    def parse_pattern_features(pattern: str) -> Dict[str, Any]:
        """
        Convert glob/regex pattern to structured features.

        Examples:
            "**/*.nii.gz"                        -> extension=".nii.gz", path_keywords=[]
            "**/anat_mprage_anonymized/*.nii.gz" -> extension=".nii.gz",
                                                    path_keywords=["anat","mprage","anonymized"]
            "VHM.*-Head.*\\.dcm"                 -> extension=".dcm", filename_prefix="vhm"
        """
        features = {
            "type": "unknown",
            "path_keywords": [],
            "filename_prefix": None,
            "filename_keywords": [],
            "extension": None,
            "exclude_keywords": []
        }

        if "**" in pattern:
            features["type"] = "path_pattern"

            # Split on "**" to get the segments between wildcards.
            # e.g. "**/anat_mprage_anonymized/*.nii.gz"
            #   -> ["", "/anat_mprage_anonymized/", ".nii.gz"]  (after split on **)
            # We then look for:
            #   - A segment ending with *.EXT  -> extract extension (with leading dot)
            #   - A segment containing a directory name -> extract keywords
            segments = pattern.split("**")

            for seg in segments:
                seg = seg.strip("/")
                if not seg:
                    continue

                # Case 1: "*.nii.gz" or "*.dcm" — extension pattern
                if seg.startswith("*."):
                    # Keep the leading dot so endswith() works correctly
                    features["extension"] = seg[1:]   # e.g. ".nii.gz"

                # Case 2: directory segment like "anat_mprage_anonymized"
                # or "anat_mprage_anonymized/*.nii.gz"
                else:
                    # Strip any trailing "/*.ext" part first
                    dir_part = seg.split("/")[0] if "/" in seg else seg
                    if dir_part and not dir_part.startswith("*"):
                        # Split on underscores and hyphens to get keywords
                        words = re.split(r'[_\-]', dir_part)
                        for word in words:
                            # Keep only alphabetic words of length >= 3
                            word_clean = re.sub(r'[^A-Za-z]', '', word)
                            if len(word_clean) >= 3:
                                features["path_keywords"].append(word_clean.lower())

                    # Also check if there is an extension in a trailing "/*.ext"
                    if "/" in seg:
                        trailing = seg.split("/")[-1]
                        if trailing.startswith("*."):
                            features["extension"] = trailing[1:]  # e.g. ".nii.gz"

        else:
            # Filename-based regex/glob pattern (no **)
            features["type"] = "filename_pattern"

            # Extract filename prefix (leading alphanumeric chars)
            if match := re.match(r'^([A-Za-z0-9]+)', pattern):
                features["filename_prefix"] = match.group(1).lower()

            # Extract filename keywords (words after delimiters)
            keywords = re.findall(r'[-_]([A-Za-z]{3,})', pattern)
            features["filename_keywords"] = [k.lower() for k in keywords]

            # Extract extension — check for escaped dot patterns first
            if r'\.dcm' in pattern or pattern.endswith('.dcm'):
                features["extension"] = ".dcm"
            elif r'\.nii\.gz' in pattern or pattern.endswith('.nii.gz'):
                features["extension"] = ".nii.gz"
            elif r'\.nii' in pattern or pattern.endswith('.nii'):
                features["extension"] = ".nii"
            elif r'\.snirf' in pattern or pattern.endswith('.snirf'):
                features["extension"] = ".snirf"
            elif r'\.mat' in pattern or pattern.endswith('.mat'):
                features["extension"] = ".mat"

        return features

    @staticmethod
    def match_file(file_relpath: str, pattern_features: Dict[str, Any]) -> bool:
        """
        Match file using structured features - completely regex-free!
        
        Args:
            file_relpath: Full relative path of file
            pattern_features: Features from parse_pattern_features()
        
        Returns:
            True if file matches features
        """
        filepath_lower = file_relpath.lower()
        filename = file_relpath.split('/')[-1].lower()
        
        # Check extension (must match)
        if pattern_features.get("extension"):
            if not filename.endswith(pattern_features["extension"]):
                return False
        
        # Check path keywords (ALL must exist)
        for keyword in pattern_features.get("path_keywords", []):
            if keyword not in filepath_lower:
                return False
        
        # Check filename prefix
        if pattern_features.get("filename_prefix"):
            if not filename.startswith(pattern_features["filename_prefix"]):
                return False
        
        # Check filename keywords (ALL must exist)
        for keyword in pattern_features.get("filename_keywords", []):
            if keyword not in filename:
                return False
        
        # Check exclude keywords (NONE should exist)
        for keyword in pattern_features.get("exclude_keywords", []):
            if keyword in filepath_lower:
                return False
        
        return True
    
    @staticmethod
    def match_files_batch(all_file_paths: List[str], patterns: List[str], 
                         exclude_patterns: Optional[List[str]] = None) -> List[str]:
        """
        Batch match files against multiple patterns.
        
        Automatically excludes duplicates (like BRIK when NIfTI exists).
        
        Args:
            all_file_paths: List of all relative file paths
            patterns: Patterns to match
            exclude_patterns: Patterns to exclude (e.g., ["**/BRIK/**"])
        
        Returns:
            List of matched file paths
        """
        # Parse pattern features
        pattern_features_list = [
            UniversalFileMatcher.parse_pattern_features(p) for p in patterns
        ]
        
        # Parse exclude patterns
        exclude_features_list = []
        if exclude_patterns:
            exclude_features_list = [
                UniversalFileMatcher.parse_pattern_features(p) for p in exclude_patterns
            ]
        
        matched = []
        
        for filepath in all_file_paths:
            # Check if excluded
            is_excluded = False
            for exclude_features in exclude_features_list:
                if UniversalFileMatcher.match_file(filepath, exclude_features):
                    is_excluded = True
                    break
            
            if is_excluded:
                continue
            
            # Check if matches any pattern
            for pattern_features in pattern_features_list:
                if UniversalFileMatcher.match_file(filepath, pattern_features):
                    matched.append(filepath)
                    break
        
        return matched


class SmartFileGrouper:
    """
    Smart file grouper - groups files by subject and scan type, handles duplicates.
    
    Features:
    1. Uses FileStructureAnalyzer's subject detection
    2. Infers scan type from path
    3. Auto-deduplicates (prefers NIfTI over BRIK)
    4. Builds BIDS-compliant filenames
    """
    
    def __init__(self, analyzer: FileStructureAnalyzer):
        self.analyzer = analyzer
    
    def group_by_subject_and_scan(self, file_paths: List[str], 
                                   subject_detection: Dict) -> Dict[str, Dict]:
        """
        Group files by subject and scan type.
        
        Args:
            file_paths: List of file paths to group
            subject_detection: Result from FileStructureAnalyzer.detect_subject_identifiers()
        
        Returns:
            {
                "sub_82352_anat_anonymized": {
                    "subject_id": "82352",
                    "scan_type": "anat",
                    "processing": "anonymized",
                    "site": "Beijing",
                    "files": [list of file paths],
                    "preferred_file": "best file path",
                    "bids_filename": "sub-82352_acq-anonymized_T1w.nii.gz"
                },
                ...
            }
        """
        groups = defaultdict(lambda: {
            "subject_id": None,
            "scan_type": None,
            "processing": None,
            "site": None,
            "files": [],
            "formats": set()
        })
        
        best_candidate = subject_detection.get("best_candidate")
        
        if not best_candidate:
            # Fallback: all files to sub-01
            for filepath in file_paths:
                groups["default"]["subject_id"] = "01"
                groups["default"]["files"].append(filepath)
                groups["default"]["scan_type"] = "unknown"
                groups["default"]["processing"] = "raw"
            
            # Add preferred file selection
            for group_key, group_data in groups.items():
                group_data["preferred_file"] = self._select_preferred_file(group_data["files"])
                group_data["bids_filename"] = "sub-01_unknown.nii.gz"
            
            return dict(groups)
        
        # Extract using best candidate's regex (already escape-fixed!)
        extraction_regex = best_candidate.get("extraction_regex")
        subject_group = best_candidate.get("subject_group", 1)
        site_group = best_candidate.get("site_group")
        
        for filepath in file_paths:
            parts = filepath.split('/')
            
            # Extract subject ID using regex
            subject_id = None
            site = None
            
            if extraction_regex:
                for part in parts:
                    try:
                        match = re.search(extraction_regex, part)
                        if match and len(match.groups()) >= subject_group:
                            subject_id = match.group(subject_group)
                            if site_group and len(match.groups()) >= site_group:
                                site = match.group(site_group)
                            break
                    except re.error:
                        pass
            
            if not subject_id:
                subject_id = "01"
            
            # Infer scan info from path
            scan_type, processing = self._infer_scan_info_from_path(filepath)
            
            # Detect file format
            file_format = self._detect_file_format(filepath)
            
            # Create group key
            group_key = f"sub_{subject_id}_{scan_type}_{processing}"
            
            # Add to group
            group = groups[group_key]
            group["subject_id"] = subject_id
            group["scan_type"] = scan_type
            group["processing"] = processing
            group["site"] = site
            group["files"].append(filepath)
            group["formats"].add(file_format)
        
        # Select preferred file for each group
        for group_key, group_data in groups.items():
            group_data["preferred_file"] = self._select_preferred_file(group_data["files"])
            group_data["bids_filename"] = build_bids_filename(
                group_data["subject_id"],
                group_data["scan_type"],
                group_data["processing"]
            )
        
        return dict(groups)
    
    def _infer_scan_info_from_path(self, filepath: str) -> Tuple[str, str]:
        """
        Infer scan type and processing state from path.
        
        Returns:
            (scan_type, processing)
        """
        path_lower = filepath.lower()
        
        # Scan type
        if 'anat' in path_lower:
            scan_type = 'anat'
        elif 'func' in path_lower:
            scan_type = 'func'
        elif 'dwi' in path_lower:
            scan_type = 'dwi'
        elif 'fmap' in path_lower:
            scan_type = 'fmap'
        else:
            scan_type = 'unknown'
        
        # Processing state
        if 'anonymized' in path_lower or 'anonymised' in path_lower:
            processing = 'anonymized'
        elif 'skullstripped' in path_lower or 'skulstripped' in path_lower:
            processing = 'skullstripped'
        elif 'rest' in path_lower:
            processing = 'rest'
        elif 'normalized' in path_lower or 'normalised' in path_lower:
            processing = 'normalized'
        else:
            processing = 'raw'
        
        return scan_type, processing
    
    def _detect_file_format(self, filepath: str) -> str:
        """Detect file format from path"""
        if '/BRIK/' in filepath or '\\BRIK\\' in filepath:
            return 'BRIK'
        elif '/NIfTI/' in filepath or '\\NIfTI\\' in filepath:
            return 'NIfTI'
        elif '/nifti/' in filepath or '\\nifti\\' in filepath:
            return 'NIfTI'
        else:
            return 'unknown'
    
    def _select_preferred_file(self, files: List[str]) -> str:
        """
        Select best file from duplicates.
        
        Priority:
        1. NIfTI over BRIK
        2. Shortest path (usually the main file)
        3. First alphabetically (for determinism)
        """
        if not files:
            return None
        
        if len(files) == 1:
            return files[0]
        
        # Filter: exclude BRIK
        non_brik = [f for f in files if 'BRIK' not in f.upper()]
        if non_brik:
            files = non_brik
        
        # Filter: prefer NIfTI
        nifti_files = [f for f in files if 'NIFTI' in f.upper()]
        if nifti_files:
            files = nifti_files
        
        # Select shortest path
        return sorted(files, key=lambda x: (len(x), x))[0]


# ============================================================================
# Helper Functions
# ============================================================================

def build_bids_filename(subject_id: str, scan_type: str, processing: str, 
                       modality_suffix: Optional[str] = None) -> str:
    """
    Build BIDS-compliant filename.
    
    Args:
        subject_id: Subject ID
        scan_type: Scan type (anat, func, dwi, fmap)
        processing: Processing state (anonymized, skullstripped, rest, etc.)
        modality_suffix: Override modality suffix (default: auto-infer)
    
    Returns:
        BIDS filename (e.g., "sub-82352_acq-anonymized_T1w.nii.gz")
    """
    # BIDS subject labels must be alphanumeric. subject_id may arrive as a raw
    # folder name with underscores/hyphens (e.g. CamCAN 'Atlanta_sub06870'),
    # which triggers validator code 62. Strip to alphanumeric here so this
    # legacy builder matches assemble_bids_name's behaviour.
    subject_id = re.sub(r"[^A-Za-z0-9]", "", str(subject_id))
    parts = [f"sub-{subject_id}"]
    
    # Determine BIDS entities based on scan type
    if scan_type == "anat":
        if processing and processing != "raw":
            parts.append(f"acq-{processing}")
        parts.append(modality_suffix or "T1w")
    
    elif scan_type == "func":
        if processing == "rest":
            parts.append("task-rest")
        elif processing and processing != "raw":
            parts.append(f"task-{processing}")
        else:
            parts.append("task-unknown")
        parts.append("bold")
    
    elif scan_type == "dwi":
        if processing and processing != "raw":
            parts.append(f"acq-{processing}")
        parts.append("dwi")
    
    elif scan_type == "fmap":
        parts.append("fieldmap")
    
    else:
        parts.append("unknown")
    
    return "_".join(parts) + ".nii.gz"


def extract_subject_ids_from_paths(file_paths: List[str], extraction_regex: str, 
                                   subject_group: int, site_group: Optional[int] = None) -> List[Dict]:
    """
    Extract subject IDs from file paths using regex.
    
    CRITICAL: This function handles the regex escape fix!
    Input regex may have double backslashes from YAML, we fix it here.
    
    Args:
        file_paths: List of file paths
        extraction_regex: Regex pattern (may have \\d from YAML)
        subject_group: Which capture group is the subject ID
        site_group: Which capture group is the site (optional)
    
    Returns:
        List of subject records with IDs and metadata
    """
    # FIX: Convert YAML double backslashes to Python regex
    regex_fixed = extraction_regex.replace(r'\\d', r'\d')
    regex_fixed = regex_fixed.replace(r'\\w', r'\w')
    regex_fixed = regex_fixed.replace(r'\\s', r'\s')
    
    seen_ids = set()
    subject_records = []
    
    for filepath in file_paths:
        parts = filepath.split('/')
        
        for part in parts:
            try:
                match = re.search(regex_fixed, part)
                
                if match and len(match.groups()) >= subject_group:
                    subject_id = match.group(subject_group)
                    
                    if subject_id not in seen_ids:
                        seen_ids.add(subject_id)
                        
                        site = None
                        if site_group and len(match.groups()) >= site_group:
                            site = match.group(site_group)
                        
                        subject_records.append({
                            "subject_id": subject_id,
                            "site": site,
                            "original_dirname": match.group(0)
                        })
                    break
            except re.error as e:
                # Regex error - pattern is invalid
                break
    
    return subject_records


# ============================================================================
# Testing & Validation
# ============================================================================

def test_universal_core():
    """Comprehensive test suite for universal_core"""
    
    print("=" * 80)
    print("Testing Universal Core Engine")
    print("=" * 80)
    
    # Test 1: FileStructureAnalyzer
    print("\n[Test 1] FileStructureAnalyzer")
    print("-" * 80)
    
    test_files_camcan = [
        "Newark_sub41006/anat_mprage_anonymized/BRIK/scan_mprage_anonymized.nii.gz",
        "Newark_sub41006/anat_mprage_anonymized/NIfTI/scan_mprage_anonymized.nii.gz",
        "Newark_sub41006/anat_mprage_skullstripped/NIfTI/scan_mprage_skullstripped.nii.gz",
        "Newark_sub41006/func_rest/NIfTI/scan_rest.nii.gz",
        "Beijing_sub82352/anat_mprage_anonymized/NIfTI/scan_mprage_anonymized.nii.gz",
        "Beijing_sub82352/func_rest/NIfTI/scan_rest.nii.gz",
        "Cambridge_sub06272/anat_mprage_anonymized/NIfTI/scan_mprage_anonymized.nii.gz",
    ]
    
    analyzer = FileStructureAnalyzer(test_files_camcan)
    
    # Test directory structure analysis
    dir_struct = analyzer.analyze_directory_structure()
    print(f"✓ Directory structure analysis:")
    print(f"  Max depth: {dir_struct['max_depth']}")
    print(f"  Structure template: {dir_struct['structure_template']}")
    
    # Test subject detection
    subject_detect = analyzer.detect_subject_identifiers(user_hint=3)
    print(f"\n✓ Subject detection:")
    print(f"  Best candidate: {subject_detect['best_candidate']['pattern_display']}")
    print(f"  Detected count: {subject_detect['best_candidate']['count']}")
    print(f"  Confidence: {subject_detect['confidence']}")
    print(f"  Extraction regex: {subject_detect['best_candidate']['extraction_regex']}")
    
    assert subject_detect['best_candidate']['count'] == 3, "Should detect 3 subjects"
    assert subject_detect['confidence'] == 'high', "Should be high confidence"
    
    # Test duplicate detection
    duplicates = analyzer.detect_duplicate_filenames()
    print(f"\n✓ Duplicate file detection:")
    print(f"  Duplicate filename count: {len(duplicates)}")
    for fname, paths in list(duplicates.items())[:2]:
        print(f"  '{fname}': {len(paths)} locations")
    
    # Test 2: UniversalFileMatcher
    print("\n[Test 2] UniversalFileMatcher")
    print("-" * 80)
    
    # Test case 1: Simple extension match
    pattern1 = "**/*.nii.gz"
    features1 = UniversalFileMatcher.parse_pattern_features(pattern1)
    print(f"\nPattern: {pattern1}")
    print(f"Features: {features1}")
    
    test_path1 = "Newark_sub41006/anat_mprage_anonymized/NIfTI/scan.nii.gz"
    match1 = UniversalFileMatcher.match_file(test_path1, features1)
    print(f"Match '{test_path1}': {match1}")
    assert match1 == True, "Should match .nii.gz files"
    
    # Test case 2: Path keyword match
    pattern2 = "**/anat_mprage_anonymized/*.nii.gz"
    features2 = UniversalFileMatcher.parse_pattern_features(pattern2)
    print(f"\nPattern: {pattern2}")
    print(f"Features: {features2}")
    
    test_path2 = "Newark_sub41006/anat_mprage_anonymized/NIfTI/scan_mprage_anonymized.nii.gz"
    match2 = UniversalFileMatcher.match_file(test_path2, features2)
    print(f"Match '{test_path2}': {match2}")
    assert match2 == True, "Should match path containing anat_mprage_anonymized"
    
    test_path3 = "Newark_sub41006/anat_mprage_skullstripped/NIfTI/scan.nii.gz"
    match3 = UniversalFileMatcher.match_file(test_path3, features2)
    print(f"Match '{test_path3}': {match3}")
    assert match3 == False, "Should not match skullstripped (keyword mismatch)"
    
    # Test case 3: Filename pattern match
    pattern3 = "VHM.*-Head.*\\.dcm"
    features3 = UniversalFileMatcher.parse_pattern_features(pattern3)
    print(f"\nPattern: {pattern3}")
    print(f"Features: {features3}")
    
    test_path4 = "VHMCT1mm-Head (64).dcm"
    match4 = UniversalFileMatcher.match_file(test_path4, features3)
    print(f"Match '{test_path4}': {match4}")
    assert match4 == True, "Should match VHM*Head*.dcm"
    
    # Test 3: Batch matching with exclude
    print("\n[Test 3] Batch matching with exclusion")
    print("-" * 80)
    
    patterns = ["**/*.nii.gz"]
    exclude = ["**/BRIK/**"]
    
    matched = UniversalFileMatcher.match_files_batch(test_files_camcan, patterns, exclude)
    print(f"Matched {len(matched)} files (excluding BRIK)")
    
    brik_count = sum(1 for f in matched if 'BRIK' in f)
    assert brik_count == 0, "Should not contain BRIK files"
    
    # Test 4: SmartFileGrouper
    print("\n[Test 4] SmartFileGrouper")
    print("-" * 80)
    
    grouper = SmartFileGrouper(analyzer)
    groups = grouper.group_by_subject_and_scan(test_files_camcan, subject_detect)
    
    print(f"Group count: {len(groups)}")
    for group_key, group_data in list(groups.items())[:3]:
        print(f"\nGroup: {group_key}")
        print(f"  Subject: {group_data['subject_id']}")
        print(f"  Scan type: {group_data['scan_type']}")
        print(f"  Processing: {group_data['processing']}")
        print(f"  File count: {len(group_data['files'])}")
        print(f"  Preferred file: {group_data['preferred_file'].split('/')[-1]}")
        print(f"  BIDS filename: {group_data['bids_filename']}")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_universal_core()
