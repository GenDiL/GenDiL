#!/usr/bin/env bash

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
   echo "common.sh is a helper for versioned GenDiL migration scripts." >&2
   exit 1
fi

: "${MIGRATION_NAME:=GenDiL migration}"
: "${MIGRATION_DESCRIPTION:=Apply mechanical GenDiL source migrations.}"

MIGRATION_APPLY=0
MIGRATION_CHECK=0
MIGRATION_BACKUP=0
MIGRATION_INCLUDE_DOCS=0
MIGRATION_TARGET="."
MIGRATION_FILE_LIST=""
MIGRATION_BACKUP_LIST=""
MIGRATION_MATCHES=0
MIGRATION_LAST_MATCHES=0
MIGRATION_WARNINGS=0

migration_usage()
{
   cat <<EOF
${MIGRATION_NAME}

${MIGRATION_DESCRIPTION}

Usage:
  $(basename "$0") [options] [path]

Options:
  --apply         Edit files in place. Without this flag, the script is a dry run.
  --dry-run       Preview changes without editing files. This is the default.
  --check         Fail if any automatic migration replacements are still needed.
  --backup        With --apply, write a .bak copy before the first edit to each file.
  --include-docs  Also scan .md, .rst, and .txt files.
  -h, --help      Show this help text.

The optional path may be a file or directory. It defaults to the current directory.
EOF
}

migration_error()
{
   echo "error: $*" >&2
   exit 2
}

migration_cleanup()
{
   if [ -n "${MIGRATION_FILE_LIST:-}" ] && [ -f "$MIGRATION_FILE_LIST" ]; then
      rm -f "$MIGRATION_FILE_LIST"
   fi
   if [ -n "${MIGRATION_BACKUP_LIST:-}" ] && [ -f "$MIGRATION_BACKUP_LIST" ]; then
      rm -f "$MIGRATION_BACKUP_LIST"
   fi
}

migration_parse_args()
{
   local target_set=0

   while [ "$#" -gt 0 ]; do
      case "$1" in
         --apply)
            MIGRATION_APPLY=1
            ;;
         --dry-run)
            MIGRATION_APPLY=0
            ;;
         --check)
            MIGRATION_CHECK=1
            ;;
         --backup)
            MIGRATION_BACKUP=1
            ;;
         --include-docs)
            MIGRATION_INCLUDE_DOCS=1
            ;;
         -h|--help)
            migration_usage
            exit 0
            ;;
         --)
            shift
            while [ "$#" -gt 0 ]; do
               if [ "$target_set" -eq 1 ]; then
                  migration_error "only one target path is supported"
               fi
               MIGRATION_TARGET="$1"
               target_set=1
               shift
            done
            break
            ;;
         -*)
            migration_error "unknown option: $1"
            ;;
         *)
            if [ "$target_set" -eq 1 ]; then
               migration_error "only one target path is supported"
            fi
            MIGRATION_TARGET="$1"
            target_set=1
            ;;
      esac
      shift
   done

   if [ "$MIGRATION_APPLY" -eq 1 ] && [ "$MIGRATION_CHECK" -eq 1 ]; then
      migration_error "--apply and --check are mutually exclusive"
   fi

   if [ ! -e "$MIGRATION_TARGET" ]; then
      migration_error "target path does not exist: $MIGRATION_TARGET"
   fi

   MIGRATION_FILE_LIST=$(mktemp "${TMPDIR:-/tmp}/gendil-migration-files.XXXXXX") ||
      migration_error "could not create temporary file list"
   MIGRATION_BACKUP_LIST=$(mktemp "${TMPDIR:-/tmp}/gendil-migration-backups.XXXXXX") ||
      migration_error "could not create temporary backup list"
   trap migration_cleanup EXIT HUP INT TERM

   migration_collect_files "$MIGRATION_TARGET"
   migration_print_header
}

migration_print_header()
{
   local mode="dry-run"
   if [ "$MIGRATION_APPLY" -eq 1 ]; then
      mode="apply"
   elif [ "$MIGRATION_CHECK" -eq 1 ]; then
      mode="check"
   fi

   echo "${MIGRATION_NAME}"
   echo "Mode: ${mode}"
   echo "Target: ${MIGRATION_TARGET}"
}

migration_should_scan()
{
   local file="$1"
   local base="${file##*/}"

   case "$base" in
      CMakeLists.txt)
         return 0
         ;;
   esac

   case "$file" in
      *.h|*.hh|*.hpp|*.hxx|*.c|*.cc|*.cpp|*.cxx|*.cu|*.cuh|*.hip|*.inl|*.ipp|*.txx|*.cmake|*.sh|*.bash)
         return 0
         ;;
   esac

   if [ "$MIGRATION_INCLUDE_DOCS" -eq 1 ]; then
      case "$file" in
         *.md|*.rst|*.txt)
            return 0
            ;;
      esac
   fi

   return 1
}

migration_is_own_script_file()
{
   local file="$1"
   local script_dir
   local file_dir
   local file_abs

   if [ -z "${SCRIPT_DIR:-}" ]; then
      return 1
   fi

   script_dir=$(cd "$SCRIPT_DIR" 2>/dev/null && pwd) || return 1
   file_dir=$(cd "$(dirname "$file")" 2>/dev/null && pwd) || return 1
   file_abs="${file_dir}/$(basename "$file")"

   case "$file_abs" in
      "${script_dir}"/*)
         return 0
         ;;
   esac

   return 1
}

migration_collect_files()
{
   local target="$1"
   local file

   : > "$MIGRATION_FILE_LIST"

   if [ -f "$target" ]; then
      if migration_should_scan "$target" && ! migration_is_own_script_file "$target"; then
         printf '%s\0' "$target" >> "$MIGRATION_FILE_LIST"
      fi
      return 0
   fi

   while IFS= read -r -d '' file; do
      if migration_should_scan "$file" && ! migration_is_own_script_file "$file"; then
         printf '%s\0' "$file" >> "$MIGRATION_FILE_LIST"
      fi
   done < <(
      find "$target" \
         \( -name .git -o -name .hg -o -name .svn -o -name .cache -o \
            -name build -o -name 'build-*' -o -name 'cmake-build-*' -o \
            -name _deps -o -name third_party -o -name external -o \
            -name vendor -o -name install -o -name 'install-*' \) \
         -prune -o -type f -print0
   )
}

migration_count_literal()
{
   local old="$1"
   local file="$2"

   OLD="$old" perl -0ne '
      $count += () = /\Q$ENV{OLD}\E/g;
      END { print $count + 0; }
   ' "$file"
}

migration_count_symbol()
{
   local old="$1"
   local file="$2"

   OLD="$old" perl -0ne '
      $old = $ENV{OLD};
      $count += () = /(?<![A-Za-z0-9_])\Q$old\E(?![A-Za-z0-9_])/g;
      END { print $count + 0; }
   ' "$file"
}

migration_count_regex()
{
   local regex="$1"
   local file="$2"

   REGEX="$regex" perl -0ne '
      $regex = $ENV{REGEX};
      $count += () = /$regex/g;
      END { print $count + 0; }
   ' "$file"
}

migration_backup_file()
{
   local file="$1"

   if [ "$MIGRATION_BACKUP" -ne 1 ]; then
      return 0
   fi

   if grep -F -x -- "$file" "$MIGRATION_BACKUP_LIST" >/dev/null 2>&1; then
      return 0
   fi

   cp -p "$file" "${file}.bak" ||
      migration_error "could not create backup for $file"
   printf '%s\n' "$file" >> "$MIGRATION_BACKUP_LIST"
}

migration_report_match()
{
   local old="$1"
   local new="$2"
   local file="$3"
   local count="$4"
   local action="dry-run"

   if [ "$MIGRATION_APPLY" -eq 1 ]; then
      action="apply"
   elif [ "$MIGRATION_CHECK" -eq 1 ]; then
      action="check"
   fi

   printf '[%s] %s -> %s: %s (%s)\n' "$action" "$old" "$new" "$file" "$count"
}

migration_apply_literal()
{
   local old="$1"
   local new="$2"
   local file="$3"

   OLD="$old" NEW="$new" perl -0pi -e 's/\Q$ENV{OLD}\E/$ENV{NEW}/g' "$file" ||
      migration_error "could not update $file"
}

migration_apply_symbol()
{
   local old="$1"
   local new="$2"
   local file="$3"

   OLD="$old" NEW="$new" perl -0pi -e 's/(?<![A-Za-z0-9_])\Q$ENV{OLD}\E(?![A-Za-z0-9_])/$ENV{NEW}/g' "$file" ||
      migration_error "could not update $file"
}

migration_apply_regex()
{
   local regex="$1"
   local new="$2"
   local file="$3"

   REGEX="$regex" NEW="$new" perl -0pi -e '$regex = $ENV{REGEX}; s/$regex/$ENV{NEW}/g' "$file" ||
      migration_error "could not update $file"
}

migration_replace_literal()
{
   local old="$1"
   local new="$2"
   local file
   local count

   MIGRATION_LAST_MATCHES=0

   while IFS= read -r -d '' file; do
      count=$(migration_count_literal "$old" "$file")
      if [ "$count" -gt 0 ]; then
         MIGRATION_LAST_MATCHES=$((MIGRATION_LAST_MATCHES + count))
         MIGRATION_MATCHES=$((MIGRATION_MATCHES + count))
         migration_report_match "$old" "$new" "$file" "$count"
         if [ "$MIGRATION_APPLY" -eq 1 ]; then
            migration_backup_file "$file"
            migration_apply_literal "$old" "$new" "$file"
         fi
      fi
   done < "$MIGRATION_FILE_LIST"
}

migration_replace_symbol()
{
   local old="$1"
   local new="$2"
   local file
   local count

   MIGRATION_LAST_MATCHES=0

   while IFS= read -r -d '' file; do
      count=$(migration_count_symbol "$old" "$file")
      if [ "$count" -gt 0 ]; then
         MIGRATION_LAST_MATCHES=$((MIGRATION_LAST_MATCHES + count))
         MIGRATION_MATCHES=$((MIGRATION_MATCHES + count))
         migration_report_match "$old" "$new" "$file" "$count"
         if [ "$MIGRATION_APPLY" -eq 1 ]; then
            migration_backup_file "$file"
            migration_apply_symbol "$old" "$new" "$file"
         fi
      fi
   done < "$MIGRATION_FILE_LIST"
}

migration_replace_regex()
{
   local regex="$1"
   local new="$2"
   local label="$3"
   local file
   local count

   MIGRATION_LAST_MATCHES=0

   while IFS= read -r -d '' file; do
      count=$(migration_count_regex "$regex" "$file")
      if [ "$count" -gt 0 ]; then
         MIGRATION_LAST_MATCHES=$((MIGRATION_LAST_MATCHES + count))
         MIGRATION_MATCHES=$((MIGRATION_MATCHES + count))
         migration_report_match "$label" "$new" "$file" "$count"
         if [ "$MIGRATION_APPLY" -eq 1 ]; then
            migration_backup_file "$file"
            migration_apply_regex "$regex" "$new" "$file"
         fi
      fi
   done < "$MIGRATION_FILE_LIST"
}

migration_warn_regex()
{
   local message="$1"
   local regex="$2"
   local file
   local count

   while IFS= read -r -d '' file; do
      count=$(
         REGEX="$regex" perl -0ne '
            $regex = $ENV{REGEX};
            $count += () = /$regex/g;
            END { print $count + 0; }
         ' "$file"
      )
      if [ "$count" -gt 0 ]; then
         MIGRATION_WARNINGS=$((MIGRATION_WARNINGS + count))
         printf '[manual] %s: %s (%s)\n' "$message" "$file" "$count"
      fi
   done < "$MIGRATION_FILE_LIST"
}

migration_note()
{
   MIGRATION_WARNINGS=$((MIGRATION_WARNINGS + 1))
   printf '[manual] %s\n' "$*"
}

migration_finish()
{
   if [ "$MIGRATION_MATCHES" -eq 0 ]; then
      echo "No automatic replacements found."
   elif [ "$MIGRATION_APPLY" -eq 1 ]; then
      echo "Applied ${MIGRATION_MATCHES} automatic replacement(s)."
   elif [ "$MIGRATION_CHECK" -eq 1 ]; then
      echo "Check failed: ${MIGRATION_MATCHES} automatic replacement(s) are still needed."
   else
      echo "Dry run only: ${MIGRATION_MATCHES} automatic replacement(s) would be made. Re-run with --apply to edit files."
   fi

   if [ "$MIGRATION_WARNINGS" -gt 0 ]; then
      echo "Manual review warning(s): ${MIGRATION_WARNINGS}."
   fi

   if [ "$MIGRATION_CHECK" -eq 1 ] && [ "$MIGRATION_MATCHES" -gt 0 ]; then
      exit 1
   fi

   exit 0
}
